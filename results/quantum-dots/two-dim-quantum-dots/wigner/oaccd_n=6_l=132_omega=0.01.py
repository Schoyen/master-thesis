from oaccd_tdho import get_filename_stub, run_oaccd_tdho

params = dict(n=6, l=132, radius_length=45, num_grid_points=201, omega=0.01)
filename_stub = get_filename_stub(params)
run_oaccd_tdho(params, filename_stub)
