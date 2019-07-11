import numpy as np

from quantum_systems import OneDimensionalHarmonicOscillator
from quantum_systems.time_evolution_operators import LaserField


def get_odho_system(n=2, l=20):
    mass = 1
    omega = 0.25
    grid_length = 10
    num_grid_points = 801
    a = 0.25
    alpha = 1

    laser_frequency = 8 * omega
    E = 1
    laser_pulse = lambda t: E * np.sin(laser_frequency * t)

    odho = OneDimensionalHarmonicOscillator(
        n,
        l,
        grid_length=grid_length,
        num_grid_points=num_grid_points,
        omega=omega,
        mass=mass,
        a=a,
        alpha=alpha,
    )
    odho.setup_system()
    odho.set_time_evolution_operator(LaserField(laser_pulse))

    return odho
