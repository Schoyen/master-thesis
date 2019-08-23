import os
import sys

import tqdm
import numpy as np

from hartree_fock import TDHF
from hartree_fock.integrators import GaussIntegrator

from isborn_system import (
    get_h2_system,
    get_lih_system,
    get_co_system,
    get_time_steps,
)
from dump import dump_data


if len(sys.argv) < 4:
    print(f"Usage: sys.argv[0] <molecule> <basis> <spin>")
    sys.exit()

molecule = sys.argv[1].lower()
basis = sys.argv[2].lower()
spin = sys.argv[3].lower()

assert molecule in ["h2", "lih", "co"]
assert basis in ["sto-3g", "6-31gss"]
assert spin in ["both", "up", "down"]

spin_dependent = False
spin_direction = "both"

if spin != "both":
    spin_dependent = True
    spin_direction = spin

molecule_func = dict(h2=get_h2_system, lih=get_lih_system, co=get_co_system)[
    molecule
]
kwargs = dict(
    basis=basis, spin_dependent=spin_dependent, spin_direction=spin_direction
)

system = molecule_func(**kwargs)
time_points, num_steps = get_time_steps()

integrator = GaussIntegrator(s=3, eps=1e-6, np=np)
tdhf = TDHF(system, integrator=integrator, verbose=True)
tdhf.set_initial_conditions()

print(f"TDHF ground state energy: " + f"{tdhf.compute_energy()}")

energy = np.zeros(num_steps, dtype=np.complex128)
dipole = np.zeros(num_steps, dtype=np.complex128)

energy[0] = tdhf.compute_energy()

z = system.dipole_moment[2]
rho = tdhf.compute_one_body_density_matrix()

dipole[0] = np.trace(rho @ z)

i = 0

try:
    for i, amp in tqdm.tqdm(
        enumerate(tdhf.solve(time_points)), total=num_steps - 1
    ):
        rho = tdhf.compute_one_body_density_matrix()

        energy[i + 1] = tdhf.compute_energy()
        dipole[i + 1] = np.trace(rho @ z)
except Exception:
    print("Crash!")

    time_points = time_points[: i + 1]
    energy = energy[: i + 1]
    dipole = dipole[: i + 1]

finally:
    dump_data(time_points, energy, dipole, "tdhf", basis, molecule, spin)
