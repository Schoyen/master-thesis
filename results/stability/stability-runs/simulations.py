import os
import sys

import tqdm
import numpy as np

from coupled_cluster.ccd import OATDCCD
from coupled_cluster.ccsd import TDCCSD
from coupled_cluster.integrators import GaussIntegrator as CCGauss

from configuration_interaction import TDCISDTQ
from configuration_interaction.integrators import GaussIntegrator as CIGauss

from tdqd_tools.io_data import write_data

from systems import get_system


T_F = 10
DT = 1e-2

NUM_TIMESTEPS = int(T_F / DT + 1)
TIME_POINTS = np.linspace(0, T_F, NUM_TIMESTEPS)


PATH = os.path.join(sys.path[0], "dat")


def dump(atom, solver, quantity_name, time, quantity):
    filename = f"{atom}_{solver}_{quantity_name}_"

    write_data(os.path.join(PATH, filename + "real.dat"), time, quantity.real)

    write_data(os.path.join(PATH, filename + "imag.dat"), time, quantity.imag)


def run_tdfci(atom):
    system = get_system(atom)

    integrator = CIGauss(s=3, eps=1e-6, np=np)

    fci = TDCISDTQ(system, integrator=integrator, verbose=True)
    fci.spin_reduce_states()
    fci.compute_ground_state(k=1)
    fci.set_initial_conditions()

    energy = np.zeros(NUM_TIMESTEPS, dtype=np.complex128)
    dip = np.zeros_like(energy)
    phase = np.zeros_like(dip)

    z = system.dipole_moment[2]

    energy[0] = fci.compute_energy()
    dip[0] = np.trace(fci.compute_one_body_density_matrix() @ z)
    phase[0] = np.abs(fci.c[0]) ** 2

    i = 0

    try:
        for i, c in tqdm.tqdm(
            enumerate(fci.solve(TIME_POINTS)), total=NUM_TIMESTEPS - 1
        ):
            energy[i + 1] = fci.compute_energy()
            dip[i + 1] = np.trace(fci.compute_one_body_density_matrix() @ z)
            phase[i + 1] = np.abs(fci.c[0]) ** 2

    finally:
        dump(atom, "tdfci", "energy", TIME_POINTS[: i + 1], energy[: i + 1])
        dump(atom, "tdfci", "dip", TIME_POINTS[: i + 1], dip[: i + 1])
        dump(atom, "tdfci", "phase", TIME_POINTS[: i + 1], phase[: i + 1])


def run_oatdccd(atom):
    system = get_system(atom)

    integrator = CCGauss(s=3, eps=1e-6, np=np)

    oatdccd = OATDCCD(system, integrator=integrator, verbose=True)
    oatdccd.compute_ground_state()
    oatdccd.set_initial_conditions()

    energy = np.zeros(NUM_TIMESTEPS, dtype=np.complex128)
    dip = np.zeros_like(energy)
    phase = np.zeros_like(dip)
    norm_t2 = np.zeros_like(phase)
    norm_l2 = np.zeros_like(phase)

    t, l, C, C_tilde = oatdccd.amplitudes

    z = C_tilde @ system.dipole_moment[2] @ C

    energy[0] = oatdccd.compute_energy()
    dip[0] = np.trace(oatdccd.compute_one_body_density_matrix() @ z)

    A_ref = np.exp(t[0][0])
    A_tilde_ref = np.exp(-t[0][0]) * (
        1 - 0.25 * np.tensordot(l[0], t[1], axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    )
    phase[0] = A_tilde_ref * A_ref

    norm_t2[0] = np.linalg.norm(t[1])
    norm_l2[0] = np.linalg.norm(l[0])

    for i, amp in tqdm.tqdm(
        enumerate(oatdccd.solve(TIME_POINTS)), total=NUM_TIMESTEPS - 1
    ):
        t, l, C, C_tilde = amp

        norm_t2[i + 1] = np.linalg.norm(t[1])
        norm_l2[i + 1] = np.linalg.norm(l[0])

        z = C_tilde @ system.dipole_moment[2] @ C

        A_ref = np.exp(t[0][0])
        A_tilde_ref = np.exp(-t[0][0]) * (
            1
            - 0.25 * np.tensordot(l[0], t[1], axes=((0, 1, 2, 3), (2, 3, 0, 1)))
        )
        phase[i + 1] = A_tilde_ref * A_ref

        energy[i + 1] = oatdccd.compute_energy()
        dip[i + 1] = np.trace(oatdccd.compute_one_body_density_matrix() @ z)

    dump(atom, "oatdccd", "energy", TIME_POINTS, energy)
    dump(atom, "oatdccd", "dip", TIME_POINTS, dip)
    dump(atom, "oatdccd", "phase", TIME_POINTS, phase)
    dump(atom, "oatdccd", "norm_t2", TIME_POINTS, norm_t2)
    dump(atom, "oatdccd", "norm_l2", TIME_POINTS, norm_l2)


def run_tdccsd(atom):
    system = get_system(atom)

    integrator = CCGauss(s=3, eps=1e-6, np=np)

    tdccsd = TDCCSD(system, integrator=integrator, verbose=True)
    tdccsd.compute_ground_state()
    tdccsd.set_initial_conditions()

    energy = np.zeros(NUM_TIMESTEPS, dtype=np.complex128)
    dip = np.zeros_like(energy)
    phase = np.zeros_like(dip)
    norm_t1 = np.zeros_like(phase)
    norm_t2 = np.zeros_like(phase)
    norm_l1 = np.zeros_like(phase)
    norm_l2 = np.zeros_like(phase)

    t, l = tdccsd.amplitudes

    z = system.dipole_moment[2]

    energy[0] = tdccsd.compute_energy()
    dip[0] = np.trace(tdccsd.compute_one_body_density_matrix() @ z)

    temp = np.einsum("ai, bj -> abij", t[1], t[1])
    temp -= temp.swapaxes(2, 3)
    temp -= temp.swapaxes(0, 1)

    A_ref = np.exp(t[0][0])
    A_tilde_ref = np.exp(-t[0][0]) * (
        1
        - 0.25 * np.tensordot(l[1], t[2], axes=((0, 1, 2, 3), (2, 3, 0, 1)))
        - np.trace(l[0] @ t[1])
        + 0.125 * np.tensordot(l[1], temp, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
    )
    phase[0] = A_tilde_ref * A_ref

    norm_t1[0] = np.linalg.norm(t[1])
    norm_t2[0] = np.linalg.norm(t[2])
    norm_l1[0] = np.linalg.norm(l[0])
    norm_l2[0] = np.linalg.norm(l[1])

    i = 0

    try:
        for i, amp in tqdm.tqdm(
            enumerate(tdccsd.solve(TIME_POINTS)), total=NUM_TIMESTEPS - 1
        ):
            t, l = amp

            norm_t1[i + 1] = np.linalg.norm(t[1])
            norm_t2[i + 1] = np.linalg.norm(t[2])
            norm_l1[i + 1] = np.linalg.norm(l[0])
            norm_l2[i + 1] = np.linalg.norm(l[1])

            temp = np.einsum("ai, bj -> abij", t[1], t[1])
            temp -= temp.swapaxes(2, 3)
            temp -= temp.swapaxes(0, 1)

            A_ref = np.exp(t[0][0])
            A_tilde_ref = np.exp(-t[0][0]) * (
                1
                - 0.25
                * np.tensordot(l[1], t[2], axes=((0, 1, 2, 3), (2, 3, 0, 1)))
                - np.trace(l[0] @ t[1])
                + 0.125
                * np.tensordot(l[1], temp, axes=((0, 1, 2, 3), (2, 3, 0, 1)))
            )
            phase[i + 1] = A_tilde_ref * A_ref

            energy[i + 1] = tdccsd.compute_energy()
            dip[i + 1] = np.trace(tdccsd.compute_one_body_density_matrix() @ z)
    finally:
        dump(atom, "tdccsd", "energy", TIME_POINTS[: i + 1], energy[: i + 1])
        dump(atom, "tdccsd", "dip", TIME_POINTS[: i + 1], dip[: i + 1])
        dump(atom, "tdccsd", "phase", TIME_POINTS[: i + 1], phase[: i + 1])
        dump(atom, "tdccsd", "norm_t1", TIME_POINTS[: i + 1], norm_t1[: i + 1])
        dump(atom, "tdccsd", "norm_t2", TIME_POINTS[: i + 1], norm_t2[: i + 1])
        dump(atom, "tdccsd", "norm_l1", TIME_POINTS[: i + 1], norm_l1[: i + 1])
        dump(atom, "tdccsd", "norm_l2", TIME_POINTS[: i + 1], norm_l2[: i + 1])
