import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pint
import tqdm

from coupled_cluster.ccd import OATDCCD
from coupled_cluster.ccsd import TDCCSD
from coupled_cluster.integrators import GaussIntegrator
from tdqd_tools.io_data import write_data

from mol_setup import get_system


def store_data(mol, solver, basis, time_points, td_energies, dipole_z):
    path = os.path.join(sys.path[0], "dat")
    filename_stub = f"{solver}_{mol}_{basis}_new_"

    write_data(
        os.path.join(path, filename_stub + "energy_real.dat"),
        time_points,
        td_energies.real,
    )

    write_data(
        os.path.join(path, filename_stub + "energy_imag.dat"),
        time_points,
        td_energies.imag,
    )

    write_data(
        os.path.join(path, filename_stub + "dipole_z_real.dat"),
        time_points,
        dipole_z.real,
    )

    write_data(
        os.path.join(path, filename_stub + "dipole_z_imag.dat"),
        time_points,
        dipole_z.imag,
    )


def run_oatdccd_simulation(
    mol,
    mol_name,
    basis="aug-ccpvdz",
    num_cycles=1,
    t_end=100,
    dt=1e-2,
    cache_freq=100,
):
    ureg = pint.UnitRegistry()

    t_start = 0
    t_end = (t_end * ureg.fs).to(ureg.a_u_time).magnitude

    num_steps = int((t_end - t_start) / dt + 1)
    time_points = np.linspace(t_start, t_end, num_steps)

    system = get_system(mol, basis, num_cycles=num_cycles)

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    integrator = GaussIntegrator(s=3, eps=1e-6, np=np)

    oatdccd = OATDCCD(system, verbose=True, integrator=integrator)
    oatdccd.compute_ground_state(tol=1e-6, termination_tol=1e-6)
    oatdccd.set_initial_conditions()

    td_energies = np.zeros(num_steps, dtype=np.complex128)
    dipole_z = np.zeros_like(td_energies)

    t, l, C, C_tilde = oatdccd.amplitudes

    rho_qp = oatdccd.compute_one_body_density_matrix()
    z = C_tilde @ system.dipole_moment[2] @ C

    td_energies[0] = oatdccd.compute_energy()
    dipole_z[0] = np.trace(rho_qp @ z)

    i = 0

    try:
        for i, amp in tqdm.tqdm(
            enumerate(oatdccd.solve(time_points)), total=num_steps - 1
        ):
            t, l, C, C_tilde = amp

            td_energies[i + 1] = oatdccd.compute_energy()

            rho_qp = oatdccd.compute_one_body_density_matrix()
            z = C_tilde @ system.dipole_moment[2] @ C

            dipole_z[i + 1] = np.trace(rho_qp @ z)

            if (i + 1) % cache_freq == 0:
                # Store data every cache_freq step

                store_data(
                    mol_name,
                    "oatdccd",
                    basis,
                    time_points[: i + 1],
                    td_energies[: i + 1],
                    dipole_z[: i + 1],
                )

    except Exception as e:
        print(e)
        print("Gauss shat itself")

        store_data(
            mol_name,
            "oatdccd",
            basis,
            time_points[: i + 1],
            td_energies[: i + 1],
            dipole_z[: i + 1],
        )


def run_tdccsd_simulation(
    mol,
    mol_name,
    basis="aug-ccpvdz",
    num_cycles=1,
    t_end=100,
    dt=1e-2,
    cache_freq=100,
):
    ureg = pint.UnitRegistry()

    t_start = 0
    t_end = (t_end * ureg.fs).to(ureg.a_u_time).magnitude

    num_steps = int((t_end - t_start) / dt + 1)
    time_points = np.linspace(t_start, t_end, num_steps)

    system = get_system(mol, basis, num_cycles=num_cycles)

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    integrator = GaussIntegrator(s=3, eps=1e-6, np=np)

    tdccsd = TDCCSD(system, verbose=True, integrator=integrator)
    tdccsd.compute_ground_state(
        t_kwargs=dict(tol=1e-6), l_kwargs=dict(tol=1e-6)
    )
    tdccsd.set_initial_conditions()

    td_energies = np.zeros(num_steps, dtype=np.complex128)
    dipole_z = np.zeros_like(td_energies)

    rho_qp = tdccsd.compute_one_body_density_matrix()
    z = system.dipole_moment[2]

    td_energies[0] = tdccsd.compute_energy()
    dipole_z[0] = np.trace(rho_qp @ z)

    i = 0

    try:
        for i, amp in tqdm.tqdm(
            enumerate(tdccsd.solve(time_points)), total=num_steps - 1
        ):
            td_energies[i + 1] = tdccsd.compute_energy()

            rho_qp = tdccsd.compute_one_body_density_matrix()
            z = system.dipole_moment[2]

            dipole_z[i + 1] = np.trace(rho_qp @ z)

            if (i + 1) % cache_freq == 0:
                # Store data every cache_freq step

                store_data(
                    mol_name,
                    "tdccsd",
                    basis,
                    time_points[: i + 1],
                    td_energies[: i + 1],
                    dipole_z[: i + 1],
                )

    except Exception as e:
        print(e)
        print("Gauss shat itself")

        store_data(
            mol_name,
            "tdccsd",
            basis,
            time_points[: i + 1],
            td_energies[: i + 1],
            dipole_z[: i + 1],
        )
