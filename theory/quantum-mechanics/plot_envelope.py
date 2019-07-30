import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from tdqd_tools.io_data import write_data

path = os.path.join(sys.path[0], "dat")

sech = lambda x: 1 / np.cosh(x)

omega = 4
E = 1

T = 1 * omega

time_points = np.linspace(-30, 30, 10001)

# We've defined phi = pi / 2.
electric_field = lambda t: E * np.cos(omega * t)


def env_sine(t, T):
    env = np.zeros_like(t)

    for i, _t in enumerate(t):
        if -np.pi * T / 2 <= _t <= np.pi * T / 2:
            env[i] = np.cos(_t / T) ** 2

    return env


env_sech = lambda t, T: sech(t / T)
env_gauss = lambda t, T: np.exp(-t ** 2 / (2 * T ** 2))


for filename, env in [
    ("sine", env_sine),
    ("sech", env_sech),
    ("gauss", env_gauss),
]:
    laser = lambda t: env(t, T) * electric_field(t)

    write_data(
        os.path.join(path, "env_" + filename + ".dat"),
        time_points,
        env(time_points, T),
    )
    write_data(
        os.path.join(path, "laser_" + filename + ".dat"),
        time_points,
        laser(time_points),
    )
    plt.plot(time_points, env(time_points, T))
    plt.plot(time_points, laser(time_points))
    plt.show()
