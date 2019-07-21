from ccsd_tdho import get_filename_stub, run_ccsd_tdho

params = dict(n=2, l=132, radius_length=8, num_grid_points=101, omega=0.1)
filename_stub = get_filename_stub(params)
run_ccsd_tdho(params, filename_stub)
