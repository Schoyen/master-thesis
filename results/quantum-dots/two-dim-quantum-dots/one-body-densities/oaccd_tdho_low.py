import os
import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import TwoDimensionalHarmonicOscillator

from hartree_fock import HartreeFock
from hartree_fock.mix import DIIS
from coupled_cluster.ccd import OACCD


n = 6
l = 90
omega = 0.1
radius_length = 12
num_grid_points = 201

os.environ["QS_CACHE_TDHO"] = "1"


tdho = TwoDimensionalHarmonicOscillator(
    n,
    l,
    radius_length=radius_length,
    num_grid_points=num_grid_points,
    omega=omega,
)
tdho.setup_system(verbose=True)

# tdho.change_to_hf_basis(tolerance=1e-10, verbose=True)


hf = HartreeFock(tdho, mixer=DIIS, verbose=True)
hf.compute_ground_state(change_system_basis=True, tol=1e-7, num_vecs=10)

oaccd = OACCD(tdho, verbose=True)
oaccd.compute_ground_state(num_vecs=20)

rho = oaccd.compute_particle_density()

plt.subplot(1, 1, 1, polar=True)
plt.contourf(tdho.T, tdho.R, rho.real)
plt.show()
