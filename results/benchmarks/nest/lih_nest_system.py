import numpy as np
import pint


from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import LaserField

from tdqd_tools.io_data import write_data


class NestLaser:
    def __init__(self, f_max=3.5e12, omega=5.44, t_stop=1):
        # Note that we assume the following units:
        #
        #    [f_max] = [W/cm^2]
        #    [omega] = [eV]
        #    [t_stop] = [fs]
        #
        # This is in correspondence with the article by Nest et al.
        # Conversion to atomic units is done in the constructor.

        ureg = pint.UnitRegistry()

        self.f_max = (
            ureg.Quantity(f_max, ureg.watt / ureg.cm ** 2)
            .to(ureg.a_u_intensity)
            .magnitude
        )

        # omega = E_nu / hbar
        self.omega = (
            (ureg.Quantity(omega, ureg.eV) / ureg.hbar)
            .to(1 / ureg.a_u_time)
            .magnitude
        )

        self.t_stop = ureg.Quantity(t_stop, ureg.fs).to(ureg.a_u_time).magnitude

    def envelope(self, t):
        return np.sin(np.pi * t / self.t_stop) ** 2

    def electric_field(self, t):
        return self.f_max * np.sin(self.omega * t)

    def __call__(self, t):
        if t >= self.t_stop:
            return 0

        # We use a negative sign as we are whacking electrons
        return -self.electric_field(t) * self.envelope(t)


def get_lih_system(polarization_axis=2):
    bond_length = 3.08  # Bohr
    molecule = f"li 0.0 0.0 0.0; h 0.0 0.0 {bond_length}"
    basis = "6-31Gs"

    lih = construct_pyscf_system_rhf(
        molecule, basis, verbose=True, symmetry=False
    )

    polarization = np.zeros(3)
    polarization[polarization_axis] = 1

    lih.set_time_evolution_operator(
        LaserField(NestLaser(), polarization_vector=polarization)
    )

    return lih


def get_time_points(t_start=0, t_end=100, dt=1e-2):
    # Time units are assumed to be in femto seconds.
    ureg = pint.UnitRegistry()

    t_start = ureg.Quantity(t_start, ureg.fs).to(ureg.a_u_time).magnitude
    t_end = ureg.Quantity(t_end, ureg.fs).to(ureg.a_u_time).magnitude

    num_steps = int((t_end - t_start) / dt + 1)
    time_points = np.linspace(t_start, t_end, num_steps)

    return time_points


if __name__ == "__main__":
    ureg = pint.UnitRegistry()

    frequency = ureg.Quantity(0.2, ureg.hartree) / 1  # Unity Planck's constant
    print(frequency)
    print(ureg.Quantity(5.44, ureg.eV).to(ureg.hartree))

    t_end_laser = ureg.Quantity(1, ureg.fs)
    print(t_end_laser)
    print(t_end_laser.to(ureg.a_u_time))
    t_end = ureg.Quantity(100, ureg.fs)
    print(t_end)
    print(t_end.to(ureg.a_u_time))

    F = ureg.Quantity(3.5e12, ureg.watt / ureg.cm ** 2)
    print(F)
    print(F.to(ureg.a_u_intensity))

    F = ureg.Quantity(1, ureg.a_u_intensity)
    print(F)
    print(F.to(ureg.watt / ureg.cm ** 2))
