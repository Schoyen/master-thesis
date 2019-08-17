import os

import numpy as np
import pandas as pd

from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.ccd import OACCD


atoms = [
    "he",
    "be",
    "ne",
    "mg",
    "ar",
    "kr",
]

basis = "aug-ccpvtz"

for atom in atoms:
    print(f"Atom: {atom}")
    system = construct_pyscf_system_rhf(atom, basis)
    oaccd = OACCD(system, verbose=False)
    oaccd.compute_ground_state()
    print(f"OACCD energy: {oaccd.compute_energy()}")
