import numpy as np


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
