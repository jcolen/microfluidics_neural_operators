using SpecialFunctions
using DifferentialEquations
using HDF5
using TerminalLoggers

"""Periodic boundary condtions."""
pbc_sym(x::Float64, L::Float64) = x - L * round(x / L)

"""Basic linear elastic force."""
elastic_force(x::Float64, k::Float64) = k * x

"""Screened hydrodynamic force with wavevector q=pi/W."""
hydro_force_screened(x::Float64, q::Float64) = (q * csch(q * x))^2

"""Derivative of screened hydrodynamic force."""
hydro_force_screened_deriv(x::Float64, q::Float64) =
	-2 * q^3 * coth(q * x) * csch(q * x)^2

"""Second derivative of screened hydrodynamic force."""
hydro_force_screened_deriv2(x::Float64, q::Float64) =
	2 * q^4 * (2 + cosh(2 * q * x)) * csch(q * x)^4

"""
Compute parameters of KdV equation corresponding to screened interactions.
e ∂w/∂t + f ∂³w/∂x³ + g ∂(w²)/∂x + c ∂w/∂x = 0
"""
function compute_KdV_params_screened(W::Float64, imax::Int=200)
	q = pi / W
	e = 1.
	f = -sum(i^3 * hydro_force_screened_deriv(float(i), q) / 3 for i in 1:imax)
	g = -sum(i^2 * hydro_force_screened_deriv2(float(i), q) for i in 1:imax)
	c = sum(-2 * i * hydro_force_screened_deriv(float(i), q) for i in 1:imax)
	return e, f, g, c
end

"""
Compute forces with periodic boundary conditions.
First parameter for ODEProblem.
"""
function compute_forces_screened_per!(du::Vector{Float64}, u::Vector{Float64},
		p::NamedTuple, t::Float64)
    du .= 0.
    for i = 1:length(u), j = 1:(i-1)
        f = hydro_force_screened(pbc_sym(u[j] - u[i], p.L), p.q)
        du[i] += f
        du[j] += f
    end
    for i = 1:(length(u) - 1)
        x = u[i+1] - u[i] - 1.
        fe = elastic_force(x, p.k)
        du[i] += fe
        du[i+1] -= fe
    end
    x = p.L + u[1] - u[end] - 1.
    fe = elastic_force(x, p.k)
    du[end] += fe
    du[1] -= fe
    nothing
end

function run_simu_screened_per(pos0::Vector{Float64}, W::Float64, k::Float64,
		                       tmax::Float64, nt::Int, algo=AutoVern7(Tsit5()),
		                       dtmax::Float64=0.1)
    # Assume that the periodic distance between
    # first and last particles is one
    L = pos0[end] - pos0[1] + 1.
    tspan = (0., tmax)
    teval = range(0., tmax, nt)
    params = (L=L, q=pi/W, k=k)

    prob = ODEProblem(compute_forces_screened_per!, pos0, tspan, params)
    @time sol = solve(prob, algo, saveat=teval, dtmax=dtmax)
    return sol
end

"""One-soliton solution of KdV equation, assuming e*f > 0."""
function one_soliton_KdV(N::Int, e::Float64, f::Float64, g::Float64,
		V0::Float64, x0::Float64)
    ampl = 3. * sqrt(V0) * sqrt(f / e) * e / g
    width = 1. / (sqrt(e / f) * sqrt(V0) / 2.)
    xx = collect(0.:N-1.)
    uu = ampl * (1. .+ tanh.((xx .- x0) ./ width))
    return xx .+ uu
end

"""Two-soliton solution of KdV equation, assuming e*f > 0."""
function two_solitons_KdV(N::Int, e::Float64, f::Float64, g::Float64,
		V1::Float64, V2::Float64, x0::Float64=0., t0::Float64=0.)
    A1 = sqrt(V1)
    A2 = sqrt(V2)
    width1 = 1. / (sqrt(e / f) * A1 / 2.)
    width2 = 1. / (sqrt(e / f) * A2 / 2.)
    ampl = (3 * sqrt(f / e) * e / g) * (A1^2 - A2^2)
    x = collect(0.:N-1.)
    T1 = tanh.((x.-x0.-V1*t0) ./ width1)
    T2 = tanh.((x.-x0.-V2*t0) ./ width2)
    return x .+ ampl .* T1 ./ (A1 .- A2 .* T2 .* T1)
end

"""Integral of the sum of two Gaussians."""
function two_gaussians(N::Int, x1::Float64, x2::Float64, h1::Float64,
		h2::Float64, w1::Float64, w2::Float64)
    x = collect(0.:N-1.)
    y = x .+ (h1 * w1 * sqrt(pi / 2.)) * erf.(-(x.-x1) ./ (sqrt(2.) * w1))
    y .+= (h2 * w2 * sqrt(pi / 2.)) * erf.(-(x.-x2) ./ (sqrt(2.) * w2))
    return y
end

"""Initial condition from dictionary of arguments."""
function init(args::Dict)
	if args["init"] == "onesoliton"
		e, f, g, c = compute_KdV_params_screened(args["W"])
		u0 = one_soliton_KdV(args["N"], e, f, g, args["V0"], args["x0"])
	elseif args["init"] == "twosolitons"
		e, f, g, c = compute_KdV_params_screened(args["W"])
		u0 = two_solitons_KdV(args["N"], e, f, g, args["V1"], args["V2"],
							  args["x0"], args["t0"])
	elseif args["init"] == "twogaussians"
		u0 = two_gaussians(args["N"], args["x1"], args["x2"], args["h1"],
						   args["h2"], args["w1"], args["w2"])
	end
	return u0
end

"""Run a simulation with given arguments and initial condition."""
run(args::Dict, u0::Vector{Float64}) = 
	run_simu_screened_per(u0, args["W"], args["k"], args["tmax"], args["nt"])

"""Run the solution to a HDF5 file."""
function save(args::Dict, sol::ODESolution, fname::String)
	h5open(fname, "w") do file
		group = create_group(file, "sol")

		group["t"] = sol.t
		group["u", chunk=(10, args["N"]), shuffle=(), deflate=3] = 
			permutedims(reduce(hcat,sol.u)) # stack in more recent versions
		for (a, v) in args
			attributes(file)[a] = v
		end

		e, f, g, c = compute_KdV_params_screened(args["W"])
		group2 = create_group(file, "KdV")
		attributes(group2)["e"] = e
		attributes(group2)["f"] = f
		attributes(group2)["g"] = g
		attributes(group2)["c"] = c
	end
	nothing
end

"""Initialize, run and save a simulation."""
function full_run(args::Dict, fname::String)
	u0 = init(args)
	sol = run(args, u0)
	save(args, sol, fname)
end
