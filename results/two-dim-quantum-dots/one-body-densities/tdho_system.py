import os
import numpy as np

from quantum_systems import TwoDimensionalHarmonicOscillator

os.environ["QS_CACHE_TDHO"] = "1"


def get_tdho(n=2, l=12):
    radius = 3
    num_grid_points = 201

    tdho = TwoDimensionalHarmonicOscillator(n, l, radius, num_grid_points)
    tdho.setup_system(verbose=True)

    return tdho
