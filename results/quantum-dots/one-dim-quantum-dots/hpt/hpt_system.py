import numpy as np

from quantum_systems import ODQD
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import HOPotential
from quantum_systems.time_evolution_operators import LaserField


class HPTLaser:
    def __init__(self, f=10, omega=0.1):
        self.f = f
        self.omega = omega

    def intensity(self, t):
        wt = self.omega * t

        if 0 <= wt <= 2 * np.pi:
            return wt * self.f / (2 * np.pi)
        elif 2 * np.pi <= wt <= 4 * np.pi:
            return self.f
        elif 4 * np.pi <= wt <= 6 * np.pi:
            return (3 - wt / (2 * np.pi)) * self.f

        return 0

    def __call__(self, t):
        f = self.intensity(t)

        return f * np.sin(self.omega * t)


def get_hpt_system(n=2, l=40, omega=1, grid_length=10, num_grid_points=401):
    odqd = ODQD(n, l, grid_length, num_grid_points)
    odqd.setup_system(potential=HOPotential(omega=omega))

    laser_pulse = HPTLaser()

    odqd.set_time_evolution_operator(LaserField(laser_pulse))

    return odqd


def get_time_points(t_start=0, t_end=20, dt=1e-2):
    num_time_points = int((t_end - t_start) / dt + 1)

    return np.linspace(t_start, t_end, num_time_points)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = get_time_points(t_end=300)
    laser = np.vectorize(HPTLaser(omega=8))

    plt.plot(t, laser(t))
    plt.show()
