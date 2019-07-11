import numpy as np
import pint

from quantum_systems import construct_pyscf_system_ao


def get_h2_system():
    ureg = pint.UnitRegistry()
    R_e = ureg.Quantity(0.7354, ureg.angstrom)
    print(R_e.to("bohr"))
    print(dir(R_e.to("bohr") / 2))

    molecule = "h 0.0 0.0 -0.6945; h 0.0 0.0 0.6945"


if __name__ == "__main__":
    get_h2_system()
