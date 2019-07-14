import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from miyagi_system import get_miyagi_system
from tdqd_tools.io_data import write_data


path = os.path.join(sys.path[0], "dat")


l = 40
odbe = get_miyagi_system(n=4, l=l)

write_data(
    os.path.join(path, "miyagi_potential.dat"),
    odbe.grid,
    odbe.potential(odbe.grid),
)

for i in range(l // 2):
    write_data(
        os.path.join(path, f"spf_l={i + 1}_sq.dat"),
        odbe.grid,
        np.abs(odbe.spf[i * 2]) ** 2 + odbe.eigen_energies[i],
    )
