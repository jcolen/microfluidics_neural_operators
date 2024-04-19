import numpy as np
from scipy.io import loadmat
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from train_Burgers import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--kernel_lr', type=float, default=1e-2)
    parser.add_argument('--beta', type=float, default=1e-7)
    parser.add_argument('--size', type=int, nargs=2, default=[50, 100])
    parser.add_argument('--pad', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_data', choices=['data/kdv_dedalus.npy', 'data/kdvb_dedalus.npy', 'data/burgers_dedalus.npy'],
                                        default='data/burgers_dedalus.npy')
    parser.add_argument('--device', choices=['cuda:0', 'cuda:1', 'cpu'], default='cuda:0') 
    hparams = vars(parser.parse_args())

    u = np.load(hparams['train_data'])
    device = torch.device(hparams['device'])

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

    save_name = os.path.basename(hparams['train_data'])[:-4]
    save_path = f'data/tb_logs/dedalus_run/{save_name}_GFNN_b{hparams["beta"]:.0e}.ckpt'

    for epoch in range(hparams['epochs']):
        base_loss = 0.
        kern_loss = 0.
        
        for data in loader:
            #Add some random noise to prevent focus on microscopic perturbations
            def add_noise(x, noise=0.025):
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
