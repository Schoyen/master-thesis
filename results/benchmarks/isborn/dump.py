import os
import sys

from tdqd_tools.io_data import write_data


def dump_data(time_points, energy, dipole, solver, basis, molecule, spin):
    path = os.path.join(sys.path[0], "dat")
    filename = f"{solver}_{molecule}_{basis}_{spin}_"

    write_data(
        os.path.join(path, filename + "energy_real.dat"),
        time_points,
        energy.real,
    )

    write_data(
        os.path.join(path, filename + "energy_imag.dat"),
        time_points,
        energy.imag,
    )

    write_data(
        os.path.join(path, filename + "dipole_z_real.dat"),
        time_points,
        dipole.real,
    )

    write_data(
        os.path.join(path, filename + "dipole_z_imag.dat"),
        time_points,
        dipole.imag,
    )
