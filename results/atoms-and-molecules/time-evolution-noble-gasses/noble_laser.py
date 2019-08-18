import pint
import numpy as np


class NobleLaser:
    def __init__(self, E=3.5e14, wavelength=200, num_cycles=1):
        # Note that we assume the following units:
        #
        #    [E] = [W/cm^2]
        #    [wavelength] = [nm]
        #
        # This is similar to the article by T. Sato
        # https://doi.org/10.1103/PhysRevA.94.023405
        ureg = pint.UnitRegistry()

        self.E = (E * ureg.watt / ureg.cm ** 2).to(ureg.a_u_intensity)
        self.E = self.E.magnitude

        self.wavelength = wavelength * ureg.nm
        self.num_cycles = num_cycles

        self.omega = (2 * np.pi * ureg.c / self.wavelength).to(
            1 / ureg.a_u_time
        )
        self.omega = self.omega.magnitude

        self.cycle = 2 * np.pi / self.omega
        self.tau = self.num_cycles * self.cycle

    def envelope(self, t):
        return np.sin(np.pi * t / self.tau) ** 2

    def __call__(self, t):
        if t >= self.tau:
            return 0

        return self.E * np.sin(self.omega * t) * self.envelope(t)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ureg = pint.UnitRegistry()
    t_end = (20 * ureg.fs).to(ureg.a_u_time)

    laser = np.vectorize(NobleLaser(num_cycles=3))

    t = np.linspace(0, t_end, 10001)

    plt.plot(t, laser(t))
    plt.show()
