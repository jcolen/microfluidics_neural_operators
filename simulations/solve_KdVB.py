import h5py
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


"""Load data from simulation file."""
def load_hdf5_simu(fname):
    with h5py.File(fname, 'r') as f:
        args = dict(f.attrs)
        kdv_args = dict(f['KdV'].attrs)
        sol = f['sol']
        tt = np.asarray(sol['t'])
        uu = np.asarray(sol['u'])
    return args, kdv_args, tt, uu


"""Deformation induced by a 2-soliton solution of the KdV equation."""
def two_solitons_KdV_deform(x, e, f, g, V1, V2, x0=0, t0=0):
    A1 = np.sqrt(V1)
    A2 = np.sqrt(V2)
    width1 = 1. / (np.sqrt(e / f) * A1 / 2.)
    width2 = 1. / (np.sqrt(e / f) * A2 / 2.)
    ampl = (3 * e / g / 2) * (A1**2 - A2**2)
    X1 = (x-x0-V1*t0) / width1
    X2 = (x-x0-V2*t0) / width2
    num = A1**2 * np.cosh(X2)**2 + A2**2 * np.sinh(X1)**2
    den = A1 * np.cosh(X1) * np.cosh(X2) - A2 * np.sinh(X1) * np.sinh(X2)
    return ampl * num / den**2


"""Deformation induced by two gaussian perturbations."""
def two_gaussians_deform(x, x1, x2, h1, h2, w1, w2):
    return h1 * np.exp(-(x-x1)**2/(2*w1**2)) \
            + h2 * np.exp(-(x-x2)**2/(2*w2**2))


"""Solve KdVB equation using Dedalus"""
def solve_Dedalus(equation, initial_cond_fun, Lx=50, Nx=1024, dealias=3/2,
                  stop_sim_time=10, timestep=5e-3, timestepper=d3.SBDF2,
                  dtype = np.float64, n_it_save=20, n_it_log=1000):
    # Bases
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)
    xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)

    # Fields
    u = dist.Field(name='u', bases=xbasis)

    # Substitutions
    dx = lambda A: d3.Differentiate(A, xcoord)

    # Problem
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation(equation)

    # Initial conditions
    x = dist.local_grid(xbasis)
    u['g'] = initial_cond_fun(x)

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Main loop
    u.change_scales(1)
    u_list = [np.copy(u['g'])]
    t_list = [solver.sim_time]
    while solver.proceed:
        solver.step(timestep)
        if solver.iteration % n_it_log == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e'
                        %(solver.iteration, solver.sim_time, timestep))
        if solver.iteration % n_it_save == 0:
            u.change_scales(1)
            u_list.append(np.copy(u['g']))
            t_list.append(solver.sim_time)

    return x, np.array(t_list), np.array(u_list)


"""Run solve_Dedalus for given arguments."""
def run(args):
    KdVB = "{}*dt(u) + {}*dx(dx(dx(u))) + {}*dx(dx(u)) = {}*u*dx(u)".format(
            args["e"], args["f"], -args["k"]/args['e'], 2.*args["g"])

    if args["init"] == "twosolitons":
        init_cond_fun = lambda x: two_solitons_KdV_deform(
                x, args["e"], args["f"], -args["g"], V1=args["V1"],
                V2=args["V2"], x0=args["x0"], t0=args["t0"])
    elif args["init"] == "twogaussians":
        init_cond_fun = lambda x: two_gaussians_deform(
                x, x1=args["x1"], x2=args["x2"], h1=args["h1"], h2=args["h2"],
                w1=args["w1"], w2=args["w2"])

    xx, tt, uu = solve_Dedalus(KdVB, init_cond_fun, Lx=args["N"],
                               Nx=args['npts'],
                               stop_sim_time=args['tmax'],
                               timestep=args['dt'],
                               n_it_save=args['n_it_save'])
    return xx, tt, uu


"""Run the solution given a simulation file and save it in the same file."""
def full_run_from_simu(fname, dt=0.1, npts=1024):
    args, kdv_args, tt, uu = load_hdf5_simu(fname)
    args2 = {**args, **kdv_args}
    args2['dt'] = dt
    args2['npts'] = npts
    args2['n_it_save'] = int(args["tmax"] / (args["nt"] - 1) / dt)
    args2['init'] = str(args2['init'], 'utf-8')

    xx, tt, uu = run(args2)

    with h5py.File(fname, 'r+') as f:
        if "dedalus/x" in f:
            del f['/dedalus/x']
        if "dedalus/t" in f:
            del f['/dedalus/t']
        if "dedalus/u" in f:
            del f['/dedalus/u']
        f.create_dataset("dedalus/x", data=xx)
        f.create_dataset("dedalus/t", data=tt)
        f.create_dataset("dedalus/u", data=uu)
