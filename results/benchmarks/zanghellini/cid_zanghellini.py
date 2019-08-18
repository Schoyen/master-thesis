import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from odho_zanghellini_system import get_odho_system

from configuration_interaction import TDCID
from configuration_interaction.integrators import GaussIntegrator

from tdqd_tools.io_data import write_data

path = "dat"

omega = 0.25
laser_frequency = 8 * omega

odho = get_odho_system()
integrator = GaussIntegrator(s=3, eps=1e-6, np=np)
tdcid = TDCID(odho, integrator=integrator, verbose=True)
tdcid.compute_ground_state()
tdcid.set_initial_conditions()

rho_tdcid = tdcid.compute_particle_density()
write_data(os.path.join(path, "rho_tdcid_real.dat"), odho.grid, rho_tdcid.real)


t_start = 0
t_end = 4 * 2 * np.pi / laser_frequency
dt = 1e-2

num_timesteps = int((t_end - t_start) / dt + 1)
time_points = np.linspace(t_start, t_end, num_timesteps)
overlap = np.zeros(num_timesteps, dtype=np.complex128)
overlap[0] = tdcid.compute_time_dependent_overlap()

for i, amp in tqdm.tqdm(
    enumerate(tdcid.solve(time_points)), total=num_timesteps - 1
):
    overlap[i + 1] = tdcid.compute_time_dependent_overlap()

write_data(
    os.path.join(path, "overlap_tdcid_real.dat"),
    time_points * laser_frequency / (2 * np.pi),
    overlap.real,
)
