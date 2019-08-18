import numpy as np

from quantum_systems import ODQD
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import HOPotential
from quantum_systems.time_evolution_operators import LaserField


def get_odho_system(n=2, l=20):
    omega = 0.25
    grid_length = 10
    num_grid_points = 801
    a = 0.25
    alpha = 1

    laser_frequency = 8 * omega
    E = 1
    laser_pulse = lambda t: E * np.sin(laser_frequency * t)

    odho = ODQD(
        n,
        l,
        grid_length=grid_length,
        num_grid_points=num_grid_points,
        a=a,
        alpha=alpha,
    )
    odho.setup_system(potential=HOPotential(omega))
    odho.set_time_evolution_operator(LaserField(laser_pulse))

    return odho
