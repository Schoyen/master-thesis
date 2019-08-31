import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from isborn_system import get_h2_system

h2_both = get_h2_system(spin_dependent=False, basis="6-31gss")
h2_up = get_h2_system(spin_dependent=True, basis="6-31gss")

plt.subplot(2, 1, 1)
plt.imshow(h2_both.dipole_moment[2].real)
plt.subplot(2, 1, 2)
plt.imshow(h2_up.dipole_moment[2].real)
plt.show()

dip_up = h2_up.dipole_moment[2].real
dip_up = np.abs(dip_up) / np.abs(dip_up).max()

dip_both = h2_both.dipole_moment[2].real
dip_both = np.abs(dip_both) / np.abs(dip_both).max()

plt.subplot(2, 1, 1)
plt.imshow(dip_both)
plt.subplot(2, 1, 2)
plt.imshow(dip_up)
plt.show()

path = os.path.join(sys.path[0], "dat")
filename = "h2_6-31gss_dip_mat"
file_up = os.path.join(path, filename + "_up.dat")
file_both = os.path.join(path, filename + "_both.dat")

with open(file_up, "w") as f_up, open(file_both, "w") as f_both:
    for ind in np.ndindex(h2_both.dipole_moment[2].shape):
        f_up.write(f"{ind[0]} {ind[1]} {dip_up[ind]}\n")
        f_both.write(f"{ind[0]} {ind[1]} {dip_both[ind]}\n")
