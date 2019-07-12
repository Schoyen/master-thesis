import os
import sys
import numpy as np
import tqdm

from miyagi_system import get_miyagi_system
from hartree_fock import TDHF
from hartree_fock.integrators import GaussIntegrator

from tdqd_tools.io_data import write_data

path = os.path.join(sys.path[0], "dat")

l = 20

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
write_data(os.path.join(path, "rho_tdhf_start_real.dat"), be.grid, rho.real)


t_start = 0
t_end = 331
dt = 1e-2

num_timesteps = int((t_end - t_start) / dt + 1)
time_points = np.linspace(t_start, t_end, num_timesteps)


# z = h2.dipole_moment[2]
# D = tdhf.C[:, h2.o] @ tdhf.C[:, h2.o].conj().T
#
# dipole_z[0] = np.trace(D @ z)
#
# for i, amp in tqdm.tqdm(
#    enumerate(tdhf.solve(time_points)), total=num_timesteps - 1
# ):
#    D = amp[:, h2.o] @ amp[:, h2.o].conj().T
#    dipole_z[i + 1] = np.trace(D @ z)
#
# write_data(
#    os.path.join(path, "dipole_z_tdhf_real.dat"), time_points, dipole_z.real
# )
