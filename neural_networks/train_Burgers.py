import numpy as np
from scipy.io import loadmat
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNextBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ConvNextBlock, self).__init__()
		
		self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding='same', padding_mode='replicate', groups=in_channels)
		self.conv2 = nn.Conv1d(out_channels, 4*out_channels, kernel_size=1)
		self.conv3 = nn.Conv1d(4*out_channels, out_channels, kernel_size=1)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = torch.sin(x)
		x = self.conv3(x)
		return x

class GFNN(nn.Module):
	def __init__(self, size=(50, 100), pad=25, **kwargs):
		super(GFNN, self).__init__()
		
		self.size = size
		self.pad = pad
		self.kernel = nn.Parameter(torch.empty((size[0]+pad, size[1]+pad),
											   dtype=torch.cfloat), requires_grad=True)
		nn.init.xavier_uniform_(self.kernel)
		
		self.read_in = nn.Sequential(
			ConvNextBlock(1, 64),
			ConvNextBlock(64, 64),
			nn.Conv1d(64, 1, kernel_size=1)
		)
		
		self.read_out = nn.Sequential(
			ConvNextBlock(1, 64),
			ConvNextBlock(64, 64),
			nn.Conv1d(64, 1, kernel_size=1)
		)
	
	def get_kernel(self):
		return self.kernel.exp()

	def get_kernel_loss(self):
		kernel = self.get_kernel()
		ker = (torch.conj(kernel) * kernel).abs().sum()
		#Don't penalize zero mode
		ker -= (torch.conj(kernel[0, 0]) * kernel[0, 0]).real.abs() 

		return ker
	
	def forward(self, x):
		b, t, l = x.shape

		pad_size = [t, l + self.pad]

		x = self.read_in(x)
		xq = torch.fft.fft2(x, s=pad_size)
		xq = xq * self.get_kernel()[None]
		x = torch.fft.ifft2(xq).real
		x = x[..., :-self.pad, :-self.pad] #Crop away spoiled terms from padding

		b, t, l = x.shape

		x = x.reshape([b*t, 1, l])
		x = self.read_out(x).reshape([b, t, l])
		return x



if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--base_lr', type=float, default=1e-4)
	parser.add_argument('--kernel_lr', type=float, default=1e-2)
	parser.add_argument('--beta', type=float, default=1e-8)
	parser.add_argument('--size', type=int, nargs=2, default=[50, 100])
	parser.add_argument('--pad', type=int, default=25)
	parser.add_argument('--epochs', type=int, default=1000)
	parser.add_argument('--batch_size', type=int, default=32)
	hparams = vars(parser.parse_args())

	u = np.load('data/rescaled_1d.npy')

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
	model = GFNN(size=hparams['size'], pad=hparams['pad'])
	model.to(device)
	optimizer = torch.optim.Adam([
		{'params': model.read_in.parameters(), 'lr':  hparams['base_lr']},
		{'params': model.read_out.parameters(), 'lr':  hparams['base_lr']},
		{'params': model.kernel, 'lr':  hparams['kernel_lr']},
	])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

	data = F.unfold(torch.FloatTensor(u)[None, None], hparams['size'])[0].T
	data = data.reshape([data.shape[0], *hparams['size']])

	print('Dataset size: ', data.shape)

	data.to(device)
	dataset = torch.utils.data.TensorDataset(data)
	loader = torch.utils.data.DataLoader(dataset,
		batch_size=hparams['batch_size'], shuffle=True, num_workers=2, pin_memory=True)

	best_loss = 1e10
	best_epoch = 0

	save_path = 'tb_logs/validation_run/burgers_GFNN_b%.0e.ckpt' % (hparams['beta'])

	for epoch in range(hparams['epochs']):
		base_loss = 0.
		kern_loss = 0.
		
		for data in loader:
			#Add some random noise to prevent focus on microscopic perturbations
			def add_noise(x, noise=0.01):
				rmse = x.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
				return x + torch.randn_like(x) * rmse * noise

			y0 = data[0].to(device)
			y0 = add_noise(y0)
			x0 = y0[..., 0:1, :]
			
			optimizer.zero_grad()
			y = model(x0)

			mse = F.mse_loss(y, y0)
			kernel = model.get_kernel()
			ker = (torch.conj(kernel) * kernel).real.abs().sum()
			ker -= (torch.conj(kernel[0, 0]) * kernel[0, 0]).real.abs() #Don't penalize zero mode constant offset

			base_loss += mse.item() / len(loader)
			kern_loss += ker.item() / len(loader)

			loss = mse + hparams['beta']*ker
			
			loss.backward()
			optimizer.step()
		
		outstr = 'Epoch=%d\tBase Loss=%g\tKernel Loss=%g' % (epoch, base_loss, kern_loss)
		print(outstr)

		loss = base_loss + hparams['beta'] * kern_loss
		scheduler.step(loss)
		
		if base_loss < best_loss:
			save_dict = {
				'state_dict': model.state_dict(),
				'hparams': hparams,
				'epoch': epoch,
				'loss': base_loss}

			torch.save(save_dict, save_path)
			best_loss = base_loss
