import numpy as np
import pint

from quantum_systems import construct_pyscf_system_ao
from quantum_systems.time_evolution_operators import LaserField


class LiLaser:
    def __init__(self, e_max=0.07, omega=0.1):
        self.e_max = e_max
        self.omega = omega
        self.cycle = 2 * np.pi / self.omega

    def e_m(self, t):
        if 0 <= t <= self.cycle:
            return t / self.cycle * self.e_max
        elif self.cycle <= t <= 2 * self.cycle:
            return self.e_max * np.sin(self.omega * t)
        elif 2 * self.cycle <= t <= 3 * self.cycle:
            return (3 - t / self.cycle) * self.e_max
        else:
            return 0

    def __call__(self, t):
        return -self.e_m(t) * np.sin(self.omega * t)


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
