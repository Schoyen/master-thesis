import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from tdqd_tools.io_data import write_data

path = os.path.join(sys.path[0], "dat")

sech = lambda x: 1 / np.cosh(x)

omega = 4
E = 1

phase = -np.pi / 2

T = 10 * np.pi / omega

time_points = np.linspace(-30, 30, 10001)

electric_field = lambda t: E * np.cos(omega * t + phase)


def env_sine(t, T, omega, num_cycles=1):
    env = np.zeros_like(t)

    for i, _t in enumerate(t):
        if 0 <= _t <= T:
            env[i] = np.sin(np.pi * _t / T) ** 2

    return env


def env_cosine(t, T, omega):
    env = np.zeros_like(t)

    for i, _t in enumerate(t):
        if -T / 2 <= _t <= T / 2:
            env[i] = np.cos(np.pi * _t / T) ** 2

    return env


env_sech = lambda t, T, omega: sech(np.pi * t / T)
env_gauss = lambda t, T, omega: np.exp(-(np.pi * t) ** 2 / (2 * T ** 2))


for filename, env in [
    ("sine", env_sine),
    ("cosine", env_cosine),
    ("sech", env_sech),
    ("gauss", env_gauss),
]:
    laser = lambda t: env(t, T, omega) * electric_field(t)

    write_data(
        os.path.join(path, "env_" + filename + ".dat"),
        time_points,
        env(time_points, T, omega),
    )
    write_data(
        os.path.join(path, "laser_" + filename + ".dat"),
        time_points,
        laser(time_points),
    )
    plt.plot(time_points, env(time_points, T, omega))
    plt.plot(time_points, laser(time_points))
    plt.show()
