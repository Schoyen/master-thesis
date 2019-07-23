import os
import sys
import time
import datetime

import numpy as np


from coupled_cluster.ccd import OATDCCD
from coupled_cluster.integrators import GaussIntegrator

from lih_nest_system import get_lih_system, get_time_points
from tdqd_tools.io_data import write_data

path = os.path.join(sys.path[0], "dat")


polarization_dict = dict(x=0, z=2)

if len(sys.argv) < 2:
    print("Specify polarization direction (x or z)")
    sys.exit()

polarization_direction = sys.argv[1]
polarization_axis = polarization_dict[polarization_direction]

lih = get_lih_system(polarization_axis=polarization_axis)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

integrator = GaussIntegrator(s=3, eps=1e-6, np=np)

oatdccd = OATDCCD(lih, integrator=integrator, verbose=True)
oatdccd.compute_ground_state(tol=1e-8, termination_tol=1e-8)
oatdccd.set_initial_conditions()

time_points = get_time_points()
num_steps = len(time_points)

dipole = np.zeros(num_steps, dtype=np.complex128)
energy = np.zeros(num_steps, dtype=np.complex128)

t_0 = time.time()
t, l, C, C_tilde = oatdccd.amplitudes

dipole_moment = C_tilde @ lih.dipole_moment[polarization_axis] @ C

rho = oatdccd.compute_one_body_density_matrix()

dipole[0] = np.trace(rho @ dipole_moment)
energy[0] = oatdccd.compute_energy()

time_accumulator = 0


try:
    for i, amp in enumerate(oatdccd.solve(time_points)):
        t_1 = time.time()

        time_accumulator += t_1 - t_0

        if i % 10 == 0:
            print(f"Iteration: {i} / {num_steps - 1}")

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

        t, l, C, C_tilde = amp

        dipole_moment = C_tilde @ lih.dipole_moment[polarization_axis] @ C

        rho = oatdccd.compute_one_body_density_matrix()

        dipole[i + 1] = np.trace(rho @ dipole_moment)
        energy[i + 1] = oatdccd.compute_energy()

except Exception:
    # Save data in case of crash
    print(f"Simulation crashed at i = {i}")
    time_points = time_points[: i + 1]
    dipole = dipole[: i + 1]
    energy = energy[: i + 1]


write_data(
    os.path.join(path, f"dipole_{polarization_direction}_oatdccd_real.dat"),
    time_points,
    dipole.real,
)

write_data(
    os.path.join(path, f"dipole_{polarization_direction}_oatdccd_imag.dat"),
    time_points,
    dipole.imag,
)

write_data(
    os.path.join(path, f"energy_{polarization_direction}_oatdccd_real.dat"),
    time_points,
    energy.real,
)

write_data(
    os.path.join(path, f"energy_{polarization_direction}_oatdccd_imag.dat"),
    time_points,
    energy.imag,
)
