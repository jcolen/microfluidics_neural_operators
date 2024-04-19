import h5py
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colorbar import ColorbarBase


def pbc_sym(x, L):
    return x - L * np.round(x / L)


def compute_deform(pos, L):
    return pbc_sym((np.roll(pos, -1, axis=0) - pos), L) - 1.


def load_hdf5_simu(fname):
    with h5py.File(fname, 'r') as f:
        args = dict(f.attrs)
        kdv_args = dict(f['KdV'].attrs)
        sol = f['sol']
        tt = np.asarray(sol['t'])
        uu = np.asarray(sol['u'])
    return args, kdv_args, tt, uu


def compute_density_translated_simu(tt, uu, c):
    xx = np.arange(uu.shape[0]) + 0.5
    L = 1 + (uu[-1, 0] - uu[0, 0])
    drho = 1/(1. + compute_deform(uu, L)) - 1.
    drho_t = np.array([
        np.roll(d, -int(c*t))
        for t, d in zip(tt, np.transpose(drho))
    ])
    return xx, drho_t


"""dw/dt = g d(w^2)/dt + k d^2w/dt^2"""
def solve_burgers(xx, uu0, ts, g, k):
    N  = len(xx)
    dx = xx[1] - xx[0]

    phi0 = np.exp((g / k) * np.cumsum(uu0) * dx)
    phi0_pad = np.pad(phi0, (N//2, N//2), "edge")

    xk = (np.arange(N) - N/2 - 0.5) * dx
    logphi_t = [np.log(phi0_pad)[N//2-1:-N//2+1]] + [
        np.log(
            np.convolve(phi0_pad, np.exp(-xk**2 / (4*k*t)),
                        mode="same")[N//2-1:-N//2+1]
            / np.sum(np.exp(-xk**2 / (4*k*t)))
        )
        for t in ts
    ]

    uu_Burgers = [np.real((k / g) * (p[2:]-p[:-2]) / (2 * dx))
                  for p in logphi_t]
    return uu_Burgers


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


w, h = 25, 25 # in mm
xshift = 100
ks = [0, 150, 300]
xmin, xmax = -350, 100
ymin, ymax = -0.006, 0.023

k_elast = 0.02
fname = f"data/twogaussians1_k_{k_elast}.h5"
cmap = plt.cm.Reds

args, kdv_args, tt, uu = load_hdf5_simu(fname)
init = str(args["init"], 'utf-8')
print(args, kdv_args)

with h5py.File(fname, 'r') as f:
    dd = f['dedalus']
    xx2 = np.asarray(dd['x'])
    tt2 = np.asarray(dd['t'])
    uu2 = np.asarray(dd['u'])

xx, drho_t = compute_density_translated_simu(tt, uu, kdv_args["c"])

fig = plt.figure(figsize=(2*w/25.4, 2*h/25.4))
ax = fig.add_axes((0.4, 0.4, 0.5, 0.5))
ax.invert_yaxis()
p = ax.pcolormesh(xshift - xx, tt[:301]/1000, drho_t[:301,:], shading="gouraud",
                  #cmap="Spectral_r", vmin=0., vmax=0.015)
                  cmap="seismic", vmin=-0.02, vmax=0.02)
ax.set_xlabel(r"$x-ct$")
ax.set_ylabel(r"$10^{-3}\ t$")
ax.xaxis.set_ticks([-300, -200, -100, 0])
ax.xaxis.set_ticklabels([-300, "", "", 0])
ax.set_xlim(xmin, xmax)
fig.savefig(f"figs/screened_2gaussians1_k{k_elast}_kymo_c.png", transparent=True, dpi=600)

fig = plt.figure(figsize=(2*w/25.4, 2*h/25.4))
ax = fig.add_axes((0.4, 0.4, 0.5, 0.5))

for i, t in enumerate(ks):
    ax.plot(-xx + xshift, drho_t[t, :], color=cmap((i+1) / len(ks)), lw=3)

for i, t in enumerate(ks[1:]):
    ax.plot(-xx2 + xshift, uu2[t, :], "-", color="0.85", lw=1.)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel(r"$x-ct$")
ax.xaxis.set_ticks([-300, -200, -100, 0])
ax.xaxis.set_ticklabels([-300, "", "", 0])
ax.yaxis.set_ticks([0., 0.01, 0.02])
ax.yaxis.set_ticklabels([])

fig.savefig(f"figs/screened_2gaussians1_k{k_elast}_c.pdf", transparent=True)

k_elast = 0.0
fname = f"data/twogaussians1_k_{k_elast}.h5"
cmap = plt.cm.Reds

args, kdv_args, tt, uu = load_hdf5_simu(fname)
init = str(args["init"], 'utf-8')
print(args, kdv_args)

with h5py.File(fname, 'r') as f:
    dd = f['dedalus']
    xx2 = np.asarray(dd['x'])
    tt2 = np.asarray(dd['t'])
    uu2 = np.asarray(dd['u'])

xx, drho_t = compute_density_translated_simu(tt, uu, kdv_args["c"])

w, h = 25, 25 # in mm
fig = plt.figure(figsize=(2*w/25.4, 2*h/25.4))
ax = fig.add_axes((0.4, 0.4, 0.5, 0.5))

for i, t in enumerate(ks):
    ax.plot(-xx + xshift, drho_t[t, :], color=cmap((i+1) / len(ks)), lw=3)

for i, t in enumerate(ks[1:]):
    ax.plot(-xx2 + xshift, uu2[t, :], "-", color="0.85", lw=1.)

ax.set_xlim(xmin, xmax)
ax.xaxis.set_ticks([-300, -200, -100, 0])
ax.xaxis.set_ticklabels([-300, "", "", 0])
ax.set_ylim(ymin, ymax)
ax.set_xlabel(r"$x-ct$")
ax.yaxis.set_ticks([0., 0.01, 0.02])
ax.yaxis.set_ticklabels(["0", "", "0.02"])
ax.set_ylabel(r"$\delta\rho$", labelpad=-15)

fig.savefig(f"figs/screened_2gaussians1_k{k_elast}_c.pdf", transparent=True)

k_elast = 0.2
fname = f"data/twogaussians1_k_{k_elast}.h5"
cmap = plt.cm.Reds

args, kdv_args, tt, uu = load_hdf5_simu(fname)
init = str(args["init"], 'utf-8')
print(args, kdv_args)

with h5py.File(fname, 'r') as f:
    dd = f['dedalus']
    xx2 = np.asarray(dd['x'])
    tt2 = np.asarray(dd['t'])
    uu2 = np.asarray(dd['u'])

xx, drho_t = compute_density_translated_simu(tt, uu, kdv_args["c"])

vvv = solve_burgers(xx, drho_t[0], tt[ks[1:]], kdv_args['g'], args['k'])

w, h = 25, 25 # in mm
fig = plt.figure(figsize=(2*w/25.4, 2*h/25.4))
ax = fig.add_axes((0.4, 0.4, 0.5, 0.5))

for i, t in enumerate(ks):
    ax.plot(-xx + xshift, drho_t[t, :], color=cmap((i+1) / len(ks)), lw=3)

for vv in vvv[1:]:
    ax.plot(-xx + xshift, vv, "-.", color="0.85", lw=1.)

ax.set_xlim(xmin, xmax)
ax.xaxis.set_ticks([-300, -200, -100, 0])
ax.xaxis.set_ticklabels([-300, "", "", 0])
ax.set_ylim(ymin, ymax)
ax.set_xlabel(r"$x-ct$")
ax.yaxis.set_ticks([0., 0.01, 0.02])
ax.yaxis.set_ticklabels([])

fig.savefig(f"figs/screened_2gaussians1_k{k_elast}_c.pdf", transparent=True)

fig = plt.figure(figsize=(30/25.4, 20/25.4))
ax = fig.add_axes((0.2, 0.8, 0.5, 0.1))
cmap = truncate_colormap(plt.cm.seismic, 0., 1.)
ti = [0., 0.5, 1.]
lbls = [-0.02, "", 0.02]
cbar = ColorbarBase(ax, orientation='horizontal', cmap=cmap,
                    ticks=ti, label=r"$\delta\rho$")
cbar.ax.set_xticklabels(lbls)
cbar.ax.tick_params(width=0, pad=0)
fig.savefig("figs/cbar_rho_gaussians1.pdf", transparent=True)

fig = plt.figure(figsize=(30/25.4, 50/25.4))
ax = fig.add_axes((0.2, 0.4, 0.05, 0.5))
cmap = truncate_colormap(plt.cm.Reds_r, 0., 1.-1./len(ks))
ti = [0., 0.5, 1.]
lbls = [3, 1.5, 0]
cbar = ColorbarBase(ax, orientation='vertical', cmap=cmap,
                    ticks=ti)
cbar.ax.set_yticklabels(lbls)
cbar.ax.tick_params(direction='in', length=4)
fig.text(0.15, 0.27, r"$10^{-3} t$")
fig.savefig("figs/cbar_times_gaussians1.pdf", transparent=True)

plt.show()
