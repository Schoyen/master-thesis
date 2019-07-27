import numpy as np
import matplotlib.pyplot as plt

from hpt_system import get_hpt_system, get_time_points

from hartree_fock import HartreeFock
from coupled_cluster.ccsd import TDCCSD
from coupled_cluster.integrators import GaussIntegrator

odqd = get_hpt_system()

hf = HartreeFock(odqd, verbose=True)
hf.compute_ground_state(change_system_basis=True)

tdccsd = TDCCSD(odqd, verbose=True)
tdccsd.compute_ground_state()
tdccsd.set_initial_conditions()

time_points = get_time_points()
