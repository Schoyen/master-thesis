import numpy as np

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import LaserField

from noble_laser import NobleLaser


def get_system(atom, basis, num_cycles):
    system = construct_pyscf_system_rhf(atom, basis)

    polarization_vector = np.zeros(3)
    # Negative polarization as we look at electrons
    polarization_vector[2] = -1

    system.set_time_evolution_operator(
        LaserField(
            NobleLaser(num_cycles), polarization_vector=polarization_vector
        )
    )

    return system
