import pint
import numpy as np


class IsbornLaser:
    def __init__(self, E_max=0.1, omega=0.06, num_cycles=3):
        # Units are in the blessed atomic units!

        self.E_max = E_max
        self.omega = omega
        self.num_cycles = num_cycles

        self.cycle = 2 * np.pi / self.omega
        self.t_stop = self.num_cycles * self.cycle

    def __call__(self, t):
        if t > self.t_stop:
            return 0

        return self.E_max * np.sin(self.omega * t)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ureg = pint.UnitRegistry()

    laser = IsbornLaser()

    t_stop = (15 * ureg.fs).to(ureg.a_u_time).magnitude

    t = np.linspace(0, t_stop, 10001)

    plt.plot((t * ureg.a_u_time).to(ureg.fs), np.vectorize(laser)(t))
    plt.show()
