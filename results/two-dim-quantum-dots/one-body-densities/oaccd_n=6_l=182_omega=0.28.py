from oaccd_tdho import get_filename_stub, run_oaccd_tdho

params = dict(n=6, l=182, radius_length=6, num_grid_points=101, omega=0.28)
filename_stub = get_filename_stub(params)
run_oaccd_tdho(params, filename_stub)
