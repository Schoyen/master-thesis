import numpy as np
from quantum_systems import OneDimensionalHarmonicOscillator
from quantum_systems.time_evolution_operators import LaserField
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import (
    AtomicPotential,
)


class MiyagiLaser:
    def __init__(self, F_0=0.0755, omega=0.057, T=331):
        self.F_0 = F_0
        self.omega = omega
        self.T = T

    def __call__(self, t):
        omega = self.omega
        F_0 = self.F_0
        T = self.T

        term = omega * np.sin(np.pi * t / T) * np.cos(omega * t)
        term += 2 * np.pi / T * np.cos(np.pi * t / T) * np.sin(omega * t)

        return -np.sin(np.pi * t / T) * term


def get_miyagi_system(n=4, l=30):
    # One-dimensional atomic system as in the article by Miyagi and Madsen.
    Z = n  # Number of protons
    c = 1  # Position of nucleus
    a = 1  # Shielded Columb parameter
    alpha = 1  # Shielded Columb parameter

    grid_length = 300
    num_grid_points = 5001

    potential = AtomicPotential(Za=Z, c=c)
    laser = MiyagiLaser()

    odbe = OneDimensionalHarmonicOscillator(
        n,
        l,
        grid_length=grid_length,
        num_grid_points=num_grid_points,
        a=a,
        alpha=alpha,
    )
    odbe.setup_system(potential=potential)
    odbe.set_time_evolution_operator(LaserField(laser))

    return odbe


if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as plt

    from tdqd_tools.io_data import write_data

    path = os.path.join(sys.path[0], "dat")

    t_arr = np.linspace(0, 331, 5001)
    laser_arr = np.zeros_like(t_arr)

    laser = MiyagiLaser()

    for i, t in enumerate(t_arr):
        laser_arr[i] = laser(t)

    write_data(os.path.join(path, "miyagi_laser.dat"), t_arr, laser_arr)
