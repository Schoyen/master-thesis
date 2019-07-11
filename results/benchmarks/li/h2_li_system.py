import numpy as np
import pint

from quantum_systems import construct_pyscf_system_ao
from quantum_systems.time_evolution_operators import LaserField


class LiLaser:
    def __init__(self, e_max=0.07, omega=0.1):
        self.e_max = e_max
        self.omega = omega

    def __call__(self, t):
        if 0 <= t <= 2 * np.pi / self.omega:
            return (self.omega * t / (2 * np.pi)) * self.e_max
        elif 2 * np.pi / self.omega <= t <= 4 * np.pi / self.omega:
            return self.e_max
        elif 4 * np.pi / self.omega <= t <= 6 * np.pi / self.omega:
            return (3 - self.omega * t / (2 * np.pi)) * self.e_max
        else:
            return 0


def get_h2_system():
    # The article by Li et al. specifies units in Ångström. We use pint to
    # convert to bohr.
    ureg = pint.UnitRegistry()
    R_e = ureg.Quantity(0.7354, ureg.angstrom)
    left = -(R_e.to(ureg.bohr) / 2).magnitude
    right = (R_e.to(ureg.bohr) / 2).magnitude

    molecule = f"h 0.0 0.0 {left}; h 0.0 0.0 {right}"

    # The basis used is 6-311++G(d, p) == 6-311++Gss in PySCF.
    basis = "6-311++Gss"

    system = construct_pyscf_system_ao(molecule, basis=basis)

    polarization = np.zeros(3)
    polarization[2] = 1

    system.set_time_evolution_operator(
        LaserField(LiLaser(), polarization_vector=polarization)
    )

    return system


if __name__ == "__main__":
    get_h2_system()
