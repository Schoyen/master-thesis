import os
import sys
import numpy as np
import tqdm

from miyagi_system import get_miyagi_system
from hartree_fock import TDHF
from hartree_fock.integrators import GaussIntegrator

from tdqd_tools.io_data import write_data

path = os.path.join(sys.path[0], "dat")

l = 40

be = get_miyagi_system(n=4, l=l)
integrator = GaussIntegrator(s=3, eps=1e-6, np=np)
tdhf = TDHF(be, integrator=integrator, verbose=True)
tdhf.compute_ground_state(tol=1e-5)
tdhf.set_initial_conditions()

import matplotlib.pyplot as plt

plt.plot(be.grid, be.potential(be.grid))

for i in range(l // 2):
    plt.plot(be.grid, be.spf[i * 2].real + be.eigen_energies[i])

plt.show()


rho = tdhf.compute_particle_density()
write_data(
    os.path.join(path, f"rho_tdhf_start_real_l={l}.dat"), be.grid, rho.real
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
    enumerate(tdhf.solve(time_points)), total=num_timesteps - 1
):
    if abs(time_points[i] - T_half) < tol:
        rho = tdhf.compute_particle_density()
        write_data(
            os.path.join(path, f"rho_tdhf_half_real_l={l}.dat"),
            be.grid,
            rho.real,
        )

rho = tdhf.compute_particle_density()
write_data(
    os.path.join(path, f"rho_tdhf_end_real_l={l}.dat"), be.grid, rho.real
)
