import sys

from oaccd_tdho import get_filename_stub, run_oaccd_tdho


if len(sys.argv) < 6:
    print(
        f"Usage: {sys.argv[0]} <n> <l> <radius_length> <num_grid_points> "
        + "<omega> <optional hf_tol> <optional oaccd_tol"
    )
    sys.exit()

n = int(sys.argv[1])
l = int(sys.argv[2])
radius_length = float(sys.argv[3])
num_grid_points = int(sys.argv[4])
omega = float(sys.argv[5])

hf_tol = 1e-7 if len(sys.argv) == 6 else float(sys.argv[6])
oaccd_tol = 1e-4 if len(sys.argv) == 7 else float(sys.argv[7])


params = dict(
    n=n,
    l=l,
    radius_length=radius_length,
    num_grid_points=num_grid_points,
    omega=omega,
)
filename_stub = get_filename_stub(params)
run_oaccd_tdho(params, filename_stub, hf_tol=hf_tol, oaccd_tol=oaccd_tol)
