import os
import sys

import tqdm
import numpy as np


from coupled_cluster.ccd import OATDCCD
from coupled_cluster.integrators import GaussIntegrator

from lih_nest_system import get_lih_system, get_time_points
from tdqd_tools.io_data import write_data

path = os.path.join(sys.path[0], "dat")

lih = get_lih_system()

oatdccd = OATDCCD(lih, verbose=True)
oatdccd.compute_ground_state(tol=1e-8, termination_tol=1e-8)
oatdccd.set_initial_conditions()

time_points = get_time_points(t_end=0.2)
num_steps = len(time_points)

dipole_x = np.zeros(num_steps, dtype=np.complex128)
dipole_z = np.zeros(num_steps, dtype=np.complex128)

t, l, C, C_tilde = oatdccd.amplitudes

x = C_tilde @ lih.dipole_moment[0] @ C
z = C_tilde @ lih.dipole_moment[2] @ C

rho = oatdccd.compute_one_body_density_matrix()

dipole_x[0] = np.trace(rho @ x)
dipole_z[0] = np.trace(rho @ x)


for i, amp in tqdm.tqdm(
    enumerate(oatdccd.solve(time_points)), total=len(time_points) - 1
):
    t, l, C, C_tilde = amp

    x = C_tilde @ lih.dipole_moment[0] @ C
    z = C_tilde @ lih.dipole_moment[2] @ C

    rho = oatdccd.compute_one_body_density_matrix()

    dipole_x[i + 1] = np.trace(rho @ x)
    dipole_z[i + 1] = np.trace(rho @ x)


write_data(
    os.path.join(path, "dipole_x_oatdccd_real.dat"), time_points, dipole_x.real
)

write_data(
    os.path.join(path, "dipole_z_oatdccd_real.dat"), time_points, dipole_z.real
)
