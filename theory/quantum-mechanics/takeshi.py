import numpy as np
import matplotlib.pyplot as plt

omega = 4
E = 1


def E_t(t, tau):
    e = np.zeros_like(t)
    for i, _t in enumerate(t):
        if 0 <= _t <= tau:
            e[i] = E * np.sin(omega * _t) * np.sin(np.pi * _t / tau) ** 2

    return e


time_points = np.linspace(0, 20, 1001)

tau = 10
plt.plot(time_points, E_t(time_points, tau))
plt.show()
