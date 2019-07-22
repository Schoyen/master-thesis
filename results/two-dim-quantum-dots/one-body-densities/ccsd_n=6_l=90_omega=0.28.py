from ccsd_tdho import get_filename_stub, run_ccsd_tdho

params = dict(n=6, l=90, radius_length=6, num_grid_points=101, omega=0.28)
filename_stub = get_filename_stub(params)
run_ccsd_tdho(params, filename_stub)
