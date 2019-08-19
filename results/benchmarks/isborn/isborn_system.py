import os

import pint
import numpy as np

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import LaserField

from isborn_laser import IsbornLaser


def get_h2_system(basis="sto-3g", spin_dependent=True, spin_direction="up"):
    # Bond lengths are found here:
    # https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Chemical_Bonding/Fundamentals_of_Chemical_Bonding/Chemical_Bonds/Bond_Lengths_and_Energies

    ureg = pint.UnitRegistry()
    bond_length = (74 * ureg.pm).to(ureg.bohr).magnitude
    molecule = f"h 0.0 0.0 {-bond_length / 2}; h 0.0 0.0 {bond_length / 2}"

    h2 = construct_pyscf_system_rhf(molecule, basis=basis)

    polarization_vector = np.zeros(3)
    # Negative sign for the charge of the electron
    polarization_vector[2] = -1

    h2.set_time_evolution_operator(
        LaserField(
            IsbornLaser(E_max=0.1), polarization_vector=polarization_vector
        )
    )

    if spin_dependent:
        remove_dipole_component(h2, spin_direction=spin_direction)

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    return h2


def get_lih_system(basis="sto-3g", spin_dependent=True, spin_direction="up"):
    bond_length = 3.08  # Bohr
    molecule = f"li 0.0 0.0 0.0; h 0.0 0.0 {bond_length}"

    lih = construct_pyscf_system_rhf(molecule, basis)

    polarization_vector = np.zeros(3)
    # Negative sign for the charge of the electron
    polarization_vector[2] = -1

    lih.set_time_evolution_operator(
        LaserField(
            IsbornLaser(E_max=0.01), polarization_vector=polarization_vector
        )
    )

    if spin_dependent:
        remove_dipole_component(lih, spin_direction=spin_direction)

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    return lih


def get_co_system(basis="sto-3g", spin_dependent=True, spin_direction="up"):
    ureg = pint.UnitRegistry()

    bond_length = (143 * ureg.pm).to(ureg.bohr).magnitude
    molecule = f"c 0.0 0.0 0.0; o 0.0 0.0 {bond_length}"

    co = construct_pyscf_system_rhf(molecule, basis)

    polarization_vector = np.zeros(3)
    # Negative sign for the charge of the electron
    polarization_vector[2] = -1

    co.set_time_evolution_operator(
        LaserField(
            IsbornLaser(E_max=0.01), polarization_vector=polarization_vector
        )
    )

    if spin_dependent:
        remove_dipole_component(co, spin_direction=spin_direction)

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    return co


def get_time_steps(t_start=0, t_end=50, dt=0.002):
    # Time units are in femtoseconds

    ureg = pint.UnitRegistry()

    t_start = (t_start * ureg.fs).to(ureg.a_u_time)
    t_end = (t_end * ureg.fs).to(ureg.a_u_time)
    dt = (dt * ureg.fs).to(ureg.a_u_time)

    num_steps = int((t_end - t_start) / dt + 1)
    time_points = np.linspace(t_start, t_end, num_steps)

    return time_points, num_steps


def remove_dipole_component(system, axis=2, spin_direction="up"):
    if spin_direction == "up":
        system._dipole_moment[axis, 1::2, 1::2] = 0
    else:
        system._dipole_moment[axis, ::2, ::2] = 0


if __name__ == "__main__":
    import matplotlib.pylab as plt

    from coupled_cluster.ccd import OATDCCD

    h2 = get_h2_system(basis="6-31g**")
    plt.imshow(h2.dipole_moment.real[2])
    plt.show()
    remove_dipole_component(h2)
    plt.imshow(h2.dipole_moment.real[2])
    plt.show()
    OATDCCD(h2, verbose=True).compute_ground_state()

    lih = get_lih_system(basis="6-31g**")
    plt.imshow(lih.dipole_moment.real[2])
    plt.show()
    remove_dipole_component(lih)
    plt.imshow(lih.dipole_moment.real[2])
    plt.show()
    OATDCCD(lih, verbose=True).compute_ground_state()

    co = get_co_system(basis="6-31g**")
    plt.imshow(co.dipole_moment.real[2])
    plt.show()
    remove_dipole_component(co)
    plt.imshow(co.dipole_moment.real[2])
    plt.show()
    OATDCCD(co, verbose=True).compute_ground_state()
