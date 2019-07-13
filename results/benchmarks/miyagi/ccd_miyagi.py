import os
import sys
import numpy as np
import tqdm

from miyagi_system import get_miyagi_system
from hartree_fock import HartreeFock
from coupled_cluster.ccd import OATDCCD
from coupled_cluster.integrators import GaussIntegrator

from tdqd_tools.io_data import write_data

path = os.path.join(sys.path[0], "dat")

l = 40

be = get_miyagi_system(n=4, l=l)

hf = HartreeFock(be, verbose=True)
hf.compute_ground_state(tol=1e-5)
be.change_basis(hf.C)

integrator = GaussIntegrator(s=3, eps=1e-6, np=np)
oatdccd = OATDCCD(be, integrator=integrator, verbose=True)
oatdccd.compute_ground_state(tol=1e-10, termination_tol=1e-10)
oatdccd.set_initial_conditions()


rho = oatdccd.compute_particle_density()
write_data(
    os.path.join(path, f"rho_oatdccd_start_real_l={l}.dat"), be.grid, rho.real
)


t_start = 0
t_end = 331
dt = 1e-2

num_timesteps = int((t_end - t_start) / dt + 1)
time_points = np.linspace(t_start, t_end, num_timesteps)

tol = dt / 100

T_half = t_end / 2
T = t_end

for i, amp in tqdm.tqdm(
    enumerate(oatdccd.solve(time_points)), total=num_timesteps - 1
):
    if abs(time_points[i] - T_half) < tol:
        rho = oatdccd.compute_particle_density()
        write_data(
            os.path.join(path, f"rho_oatdccd_half_real_l={l}.dat"),
            be.grid,
            rho.real,
        )

rho = oatdccd.compute_particle_density()
write_data(
    os.path.join(path, f"rho_oatdccd_end_real_l={l}.dat"), be.grid, rho.real
)
