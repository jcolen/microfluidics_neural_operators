from solve_KdVB import full_run_from_simu

full_run_from_simu("data/twosolitons_V1_0.05_V2_0.01_k_0.0.h5")
full_run_from_simu("data/twosolitons_V1_0.05_V2_0.01_k_0.02.h5")
full_run_from_simu("data/twosolitons_V1_0.05_V2_0.01_k_0.2.h5")

full_run_from_simu("data/twogaussians1_k_0.0.h5")
full_run_from_simu("data/twogaussians1_k_0.02.h5")
full_run_from_simu("data/twogaussians1_k_0.2.h5")

full_run_from_simu("data/twogaussians2_k_0.0.h5", npts=4096, dt=0.02)
full_run_from_simu("data/twogaussians2_k_0.02.h5")
full_run_from_simu("data/twogaussians2_k_0.2.h5")
