include("non_reciprocal_chain.jl")

# Fig 4
V1, V2 = 0.05, 0.01
for k = [0., 0.02, 0.2]
	println(k)
	args = Dict("N"=>500, "W"=>2.0, "tmax"=>10000., "nt"=>1001, "k"=>k,
				"init"=>"twosolitons", "V1"=>V1, "V2"=>V2, "t0"=>-1000.,
				"x0"=>200.)
	fname = "data/twosolitions_V1_$(V1)_V2_$(V2)_k_$k.h5"
	full_run(args, fname)
end

# Fig S4a
for k = [0., 0.02, 0.2]
	println(k)
	args = Dict("N"=>500, "W"=>2.0, "tmax"=>1000., "nt"=>1001, "k"=>k,
				"init"=>"twogaussians", "x1"=>100., "x2"=>130., "h1"=>0.015,
				"h2"=>-0.005, "w1"=>10., "w2"=>10.)
	fname = "data/twogaussians1_k_$k.h5"
	full_run(args, fname)
end

# Fig S4b
for k = [0., 0.02, 0.2]
	println(k)
	args = Dict("N"=>500, "W"=>2.0, "tmax"=>1000., "nt"=>1001, "k"=>k,
				"init"=>"twogaussians", "x1"=>150., "x2"=>180., "h1"=>0.15,
				"h2"=>-0.02, "w1"=>10., "w2"=>10.)
	fname = "data/twogaussians2_k_$k.h5"
	full_run(args, fname)
end
