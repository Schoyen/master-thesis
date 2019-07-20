import os
import sys
import numpy as np

from tdho_system import get_tdho
from configuration_interaction import CISD

n = 2
l = 12

tdho = get_tdho(n=n, l=l)

cisd = CISD(tdho, verbose=True)
cisd.compute_ground_state()
print(cisd.energies[0])

rho = cisd.compute_particle_density()

path = os.path.join(sys.path[0], "dat")
filename = f"cisd_n={n}_l={l}_rho_real.dat"
filename = os.path.join(path, filename)

np.savetxt(
    filename,
    np.c_[
        tdho.T.ravel()[:, np.newaxis],
        tdho.R.ravel()[:, np.newaxis],
        rho.real.ravel()[:, np.newaxis],
    ],
)


import matplotlib.pyplot as plt

fig = plt.figure()
fig.add_subplot(1, 1, 1, polar=True)

plt.contourf(tdho.T, tdho.R, rho.real)

plt.show()
