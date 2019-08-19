from ccsd_tdho import get_filename_stub, run_ccsd_tdho

params = dict(n=20, l=132, radius_length=10, num_grid_points=101, omega=0.28)
filename_stub = get_filename_stub(params)
run_ccsd_tdho(params, filename_stub)
