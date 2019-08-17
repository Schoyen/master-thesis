import os

import numpy as np
import pandas as pd

from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.ccd import OACCD


atoms = [
    "he 0.0 0.0 0.0",
    "be 0.0 0.0 0.0",
    "ne 0.0 0.0 0.0",
    "mg 0.0 0.0 0.0",
    "ar 0.0 0.0 0.0",
    "kr 0.0 0.0 0.0",
]

basis = "aug-ccpvdz"

for atom in atoms:
    print(f"Atom: {atom}")
    system = construct_pyscf_system_rhf(atom, basis)
    oaccd = OACCD(system, verbose=False)
    oaccd.compute_ground_state()
    print(f"OACCD energy: {oaccd.compute_energy()}")
