import os
import numpy as np

from quantum_systems import TwoDimensionalHarmonicOscillator

os.environ["QS_CACHE_TDHO"] = "1"


def get_tdho(*args, **kwargs):
    tdho = TwoDimensionalHarmonicOscillator(*args, **kwargs)
    tdho.setup_system(verbose=True)

    return tdho
