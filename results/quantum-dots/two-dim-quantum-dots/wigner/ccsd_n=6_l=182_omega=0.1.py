from ccsd_tdho import get_filename_stub, run_ccsd_tdho

params = dict(n=6, l=182, radius_length=12, num_grid_points=201, omega=0.1)
filename_stub = get_filename_stub(params)
run_ccsd_tdho(params, filename_stub)
