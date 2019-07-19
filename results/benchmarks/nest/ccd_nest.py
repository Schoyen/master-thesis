import os
import sys
import time

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

time_points = get_time_points()
num_steps = len(time_points)

dipole_x = np.zeros(num_steps, dtype=np.complex128)
dipole_z = np.zeros(num_steps, dtype=np.complex128)
energy = np.zeros(num_steps, dtype=np.complex128)

t_0 = time.time()
t, l, C, C_tilde = oatdccd.amplitudes

x = C_tilde @ lih.dipole_moment[0] @ C
z = C_tilde @ lih.dipole_moment[2] @ C

rho = oatdccd.compute_one_body_density_matrix()

dipole_x[0] = np.trace(rho @ x)
dipole_z[0] = np.trace(rho @ x)
energy[0] = oatdccd.compute_energy()
t_1 = time.time()

time_accumulator = 0


try:
    for i, amp in enumerate(oatdccd.solve(time_points)):

        time_accumulator += t_1 - t_0

        if i % 10:
            print(f"Iteration: {i + 1} / {num_steps} (time: {t_1 - t_0} sec)")

            time_per_iteration = time_accumulator / (i + 1)
            time_left = (num_steps - (i + 1)) * time_per_iteration
            print(
                f"""
            Total run time: {time_accumulator}
            Time per iteration: {time_per_iteration}
            Time remaining: {time_left}
            """
            )

        t_0 = time.time()
        t, l, C, C_tilde = amp

        x = C_tilde @ lih.dipole_moment[0] @ C
        z = C_tilde @ lih.dipole_moment[2] @ C

        rho = oatdccd.compute_one_body_density_matrix()

        dipole_x[i + 1] = np.trace(rho @ x)
        dipole_z[i + 1] = np.trace(rho @ x)
        energy[i + 1] = oatdccd.compute_energy()
        t_1 = time.time()

except Exception:
    # Save data in case of crash
    pass


write_data(
    os.path.join(path, "dipole_x_oatdccd_real.dat"), time_points, dipole_x.real
)

write_data(
    os.path.join(path, "dipole_x_oatdccd_imag.dat"), time_points, dipole_x.imag
)

write_data(
    os.path.join(path, "dipole_z_oatdccd_real.dat"), time_points, dipole_z.real
)

write_data(
    os.path.join(path, "dipole_z_oatdccd_imag.dat"), time_points, dipole_z.imag
)

write_data(
    os.path.join(path, "energy_oatdccd_real.dat"), time_points, energy.real
)

write_data(
    os.path.join(path, "energy_oatdccd_imag.dat"), time_points, energy.imag
)
