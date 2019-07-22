import os
import sys
import time
import datetime

import numpy as np


from coupled_cluster.ccsd import TDCCSD
from coupled_cluster.integrators import GaussIntegrator

from lih_nest_system import get_lih_system, get_time_points
from tdqd_tools.io_data import write_data

path = os.path.join(sys.path[0], "dat")

lih = get_lih_system()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

integrator = GaussIntegrator(s=3, eps=1e-6, np=np)

tdccsd = TDCCSD(lih, integrator=integrator, verbose=True)
tdccsd.compute_ground_state(t_kwargs=dict(tol=1e-8), l_kwargs=dict(tol=1e-8))
tdccsd.set_initial_conditions()

time_points = get_time_points()
num_steps = len(time_points)

dipole_x = np.zeros(num_steps, dtype=np.complex128)
dipole_z = np.zeros(num_steps, dtype=np.complex128)
energy = np.zeros(num_steps, dtype=np.complex128)

t_0 = time.time()

x = lih.dipole_moment[0]
z = lih.dipole_moment[2]

rho = tdccsd.compute_one_body_density_matrix()

dipole_x[0] = np.trace(rho @ x)
dipole_z[0] = np.trace(rho @ x)
energy[0] = tdccsd.compute_energy()

time_accumulator = 0


try:
    for i, amp in enumerate(tdccsd.solve(time_points)):
        t_1 = time.time()

        time_accumulator += t_1 - t_0

        if i % 10 == 0:
            print(f"Iteration: {i + 1} / {num_steps}")

            time_per_iteration = time_accumulator / (i + 1)
            time_left = (num_steps - (i + 1)) * time_per_iteration
            time_remaining = datetime.timedelta(seconds=time_left)
            print(
                f"""
            Total run time: {time_accumulator} sec
            Time per iteration: {time_per_iteration} sec
            Time remaining: {time_left} sec
            Time remaining: {time_remaining}
            """
            )

        t_0 = time.time()

        x = lih.dipole_moment[0]
        z = lih.dipole_moment[2]

        rho = tdccsd.compute_one_body_density_matrix()

        dipole_x[i + 1] = np.trace(rho @ x)
        dipole_z[i + 1] = np.trace(rho @ x)
        energy[i + 1] = tdccsd.compute_energy()

except Exception:
    # Save data in case of crash
    pass


write_data(
    os.path.join(path, "dipole_x_tdccsd_real.dat"), time_points, dipole_x.real
)

write_data(
    os.path.join(path, "dipole_x_tdccsd_imag.dat"), time_points, dipole_x.imag
)

write_data(
    os.path.join(path, "dipole_z_tdccsd_real.dat"), time_points, dipole_z.real
)

write_data(
    os.path.join(path, "dipole_z_tdccsd_imag.dat"), time_points, dipole_z.imag
)

write_data(
    os.path.join(path, "energy_tdccsd_real.dat"), time_points, energy.real
)

write_data(
    os.path.join(path, "energy_tdccsd_imag.dat"), time_points, energy.imag
)
