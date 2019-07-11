import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from odho_zanghellini_system import get_odho_system
from hartree_fock import TDHF
from coupled_cluster.integrators import GaussIntegrator

from tdqd_tools.io_data import write_data


path = "dat"

omega = 0.25
laser_frequency = 8 * omega

odho = get_odho_system()
integrator = GaussIntegrator(s=3, eps=1e-6, np=np)
tdhf = TDHF(odho, integrator=integrator, verbose=True)
tdhf.compute_ground_state(tol=1e-5)
tdhf.set_initial_conditions()

rho_tdhf = tdhf.compute_particle_density()
write_data(os.path.join(path, "rho_tdhf_real.dat"), odho.grid, rho_tdhf.real)

t_start = 0
t_end = 4 * 2 * np.pi / laser_frequency
dt = 1e-2

num_timesteps = int((t_end - t_start) / dt + 1)
time_points = np.linspace(t_start, t_end, num_timesteps)
overlap = np.zeros(num_timesteps, dtype=np.complex128)
overlap[0] = tdhf.compute_time_dependent_overlap()

for i, amp in tqdm.tqdm(
    enumerate(tdhf.solve(time_points)), total=num_timesteps - 1
):
    overlap[i + 1] = tdhf.compute_time_dependent_overlap()

write_data(
    os.path.join(path, "overlap_tdhf_real.dat"),
    time_points * laser_frequency / (2 * np.pi),
    overlap.real,
)
