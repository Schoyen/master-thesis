import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from odho_zanghellini_system import get_odho_system
from coupled_cluster.ccsd import TDCCSD
from coupled_cluster.integrators import GaussIntegrator

from tdqd_tools.io_data import write_data


path = "dat"

omega = 0.25
laser_frequency = 8 * omega

odho = get_odho_system()
integrator = GaussIntegrator(s=3, eps=1e-6, np=np)
tdccsd = TDCCSD(odho, integrator=integrator, verbose=True)
tdccsd.compute_ground_state(t_kwargs=dict(tol=1e-10), l_kwargs=dict(tol=1e-10))
tdccsd.set_initial_conditions()

rho_tdccsd = tdccsd.compute_particle_density()
write_data(
    os.path.join(path, "rho_tdccsd_real.dat"), odho.grid, rho_tdccsd.real
)


t_start = 0
t_end = 4 * 2 * np.pi / laser_frequency
dt = 1e-2

num_timesteps = int((t_end - t_start) / dt + 1)
time_points = np.linspace(t_start, t_end, num_timesteps)
overlap = np.zeros(num_timesteps, dtype=np.complex128)
overlap[0] = tdccsd.compute_time_dependent_overlap()

for i, amp in tqdm.tqdm(
    enumerate(tdccsd.solve(time_points)), total=num_timesteps - 1
):
    overlap[i + 1] = tdccsd.compute_time_dependent_overlap()

write_data(
    os.path.join(path, "overlap_tdccsd_real.dat"),
    time_points * laser_frequency / (2 * np.pi),
    overlap.real,
)
