import os
import sys
import numpy as np
import tqdm

from h2_li_system import get_h2_system
from hartree_fock import HartreeFock
from coupled_cluster.ccsd import TDCCSD
from coupled_cluster.integrators import GaussIntegrator

from tdqd_tools.io_data import write_data


path = os.path.join(sys.path[0], "dat")

h2 = get_h2_system()
integrator = GaussIntegrator(s=3, eps=1e-6, np=np)

hf = HartreeFock(h2, verbose=True)
hf.compute_ground_state(tol=1e-12, max_iterations=100)
h2.change_basis(hf.C)

tdccsd = TDCCSD(h2, integrator=integrator, verbose=True)
tdccsd.compute_ground_state(t_kwargs=dict(tol=1e-12), l_kwargs=dict(tol=1e-12))
tdccsd.set_initial_conditions()


t_start = 0
t_end = 225
dt = 1e-2

num_timesteps = int((t_end - t_start) / dt + 1)
time_points = np.linspace(t_start, t_end, num_timesteps)

dipole_z = np.zeros(num_timesteps, dtype=np.complex128)

z = h2.dipole_moment[2]
rho_qp = tdccsd.compute_one_body_density_matrix()
rho_qp = 0.5 * (rho_qp + rho_qp.conj().T)

dipole_z[0] = np.trace(rho_qp @ z)

for i, amp in tqdm.tqdm(
    enumerate(tdccsd.solve(time_points)), total=num_timesteps - 1
):
    rho_qp = tdccsd.compute_one_body_density_matrix()
    rho_qp = 0.5 * (rho_qp + rho_qp.conj().T)

    dipole_z[i + 1] = np.trace(rho_qp @ z)

write_data(
    os.path.join(path, "dipole_z_tdccsd_real.dat"), time_points, dipole_z.real
)
