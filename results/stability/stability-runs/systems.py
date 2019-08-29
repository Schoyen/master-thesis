import os
import sys
import numpy as np

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import LaserField
from tdqd_tools.io_data import write_data


E_HE = 100
E_BE = 1

OMEGA_HE = 2.8735643
OMEGA_BE = 0.2068175

T_D = 5

KWARGS = {
    "he": dict(E=E_HE, omega=OMEGA_HE, t_d=T_D),
    "be": dict(E=E_BE, omega=OMEGA_BE, t_d=T_D),
}


POL = np.zeros(3)
POL[2] = -1


class ThomasLaser:
    def __init__(self, E, omega, t_d):
        self.E = E
        self.omega = omega
        self.t_d = t_d

    def envelope(self, t):
        return np.sin(np.pi * t / self.t_d) ** 2

    def __call__(self, t):
        if t > self.t_d:
            return 0

        return self.E * np.cos(self.omega * t) * self.envelope(t)


def get_system(atom):
    basis = "cc-pvdz"

    system = construct_pyscf_system_rhf(atom, basis)
    system.set_time_evolution_operator(
        LaserField(ThomasLaser(**KWARGS[atom]), polarization_vector=POL)
    )

    return system


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = np.linspace(0, 10, 10001)

    laser_he = np.vectorize(ThomasLaser(E_HE, OMEGA_HE, 5))(t)
    laser_be = np.vectorize(ThomasLaser(E_BE, OMEGA_BE, 5))(t)

    path = os.path.join(sys.path[0], "dat")

    write_data(os.path.join(path, "laser_he.dat"), t, laser_he)

    write_data(os.path.join(path, "laser_be.dat"), t, laser_be)
