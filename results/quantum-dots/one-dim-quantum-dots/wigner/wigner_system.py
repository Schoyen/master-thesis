import tqdm
import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import ODQD
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import HOPotential

from configuration_interaction import CISD
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles
from hartree_fock import HartreeFock
from hartree_fock.mix import DIIS


n = 4
l = 60

omega = 0.1

grid_length = 100
num_grid_points = 401

odho = ODQD(n, l, grid_length, num_grid_points)
odho.setup_system(potential=HOPotential(omega=omega))

plt.plot(odho.grid, odho.potential(odho.grid))

for i in range(l // 2):
    plt.plot(
        odho.grid,
        odho.eigen_energies[i] + np.abs(odho.spf[2 * i]) ** 2,
        label=rf"i = {i}",
    )

plt.legend(loc="best")
plt.show()


hf = HartreeFock(odho, mixer=DIIS, verbose=True)
hf.compute_ground_state(change_system_basis=True, num_vecs=10)

ccsd = CoupledClusterSinglesDoubles(odho, verbose=True)
ccsd.compute_ground_state()


# cisd = CISD(odho, verbose=True)
# cisd.compute_ground_state(k=1)
#
# print(cisd.energies[0])

rho = ccsd.compute_particle_density().real

plt.plot(odho.grid, rho)
plt.show()
