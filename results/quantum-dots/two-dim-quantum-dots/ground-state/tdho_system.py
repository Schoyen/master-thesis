import os
import numpy as np

from quantum_systems import TwoDimensionalHarmonicOscillator
from hartree_fock import RHF
from hartree_fock.mix import DIIS

os.environ["QS_CACHE_TDHO"] = "1"


def cache_large_system(*args, **kwargs):
    tdho = TwoDimensionalHarmonicOscillator(*args, **kwargs)
    tdho.setup_system(verbose=True, add_spin=False, anti_symmetrize=False)

    return tdho


def get_tdho(*args, add_spin=True, **kwargs):
    tdho = TwoDimensionalHarmonicOscillator(*args, **kwargs)
    tdho.setup_system(verbose=True, add_spin=add_spin)

    return tdho


def get_rhf_tdho(*args, tol=1e-7, **kwargs):
    tdho = cache_large_system(*args, **kwargs)

    rhf = RHF(tdho, verbose=True, mixer=DIIS)
    rhf.compute_ground_state(change_system_basis=True)

    return rhf
