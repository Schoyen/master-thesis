import numpy as np
import pint


from quantum_systems import construct_pyscf_system
from quantum_systems.time_evolution_operators import LaserField

from tdqd_tools.io_data import write_data


class NestLaser:
    def __init__(self, f_max=0, omega=0.2, t_stop=41.34137):
        self.f_max = f_max
        self.omega = omega
        self.t_stop = t_stop

    def __call__(self, t):
        if t >= t_stop:
            return 0

        # We use a negative sign as we are whacking electrons
        return -self.f_max * np.sin(self.omega * t) ** 2


def get_lih_system():
    bond_length = 3.08  # Bohr
    molecule = f"li 0.0 0.0 0.0; h 0.0 0.0 0.0 {bond_length}"
    basis = "6-31Gs"

    lih = construct_pyscf_system(molecule, basis, verbose=True)

    omega = ureg.Quantity(0.2, ureg.hartree) / 1  # Unity Planck's constant

    t_start = 0
    t_end_laser = ureg.Quantity(1, ureg.fs)
    t_end = ureg.Quantity(100, ureg.fs)


if __name__ == "__main__":
    ureg = pint.UnitRegistry()

    frequency = ureg.Quantity(0.2, ureg.hartree) / 1  # Unity Planck's constant
    print(frequency)

    t_end_laser = ureg.Quantity(1, ureg.fs)
    print(t_end_laser)
    print(t_end_laser.to(ureg.a_u_time))
    t_end = ureg.Quantity(100, ureg.fs)
    print(t_end)
    print(t_end.to(ureg.a_u_time))

    F = ureg.Quantity(3.5e12, ureg.watt / ureg.cm ** 2)
    F = ureg.Quantity(3.51e16, ureg.watt / ureg.cm ** 2)
    print(F)
    print(F.to(ureg.a_u_intensity))

    F = ureg.Quantity(1, ureg.a_u_intensity)
    print(F)
    print(F.to(ureg.watt / ureg.cm ** 2))
