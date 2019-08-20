import os
import sys

import numpy as np
import pandas as pd

from tdho_system import get_rhf_tdho, get_tdho
from coupled_cluster.ccd import CoupledClusterDoubles as CCD, OACCD
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles as CCSD


n_list = [2, 6, 12, 20]
n_list = [20]
omega_list = [0.1, 0.28, 0.5, 1.0]
omega_list = [1.0]
l = 90

df = pd.DataFrame(columns=["n", "omega", "rhf", "ccd", "ccsd", "noccd"])


for n in n_list:
    for omega in omega_list:
        if n == 12 and omega < 0.28:
            continue

        if n == 20 and omega < 0.5:
            continue

        res_dict = dict(n=n, omega=omega)

        rhf = get_rhf_tdho(
            n, l, radius_length=40, num_grid_points=201, tol=1e-5, omega=omega
        )
        res_dict["rhf"] = rhf.compute_energy().real

        system = rhf.system
        system.change_to_spin_orbital_basis()

        # try:
        #    ccd = CCD(system, verbose=True)
        #    ccd.compute_ground_state()
        #    res_dict["ccd"] = ccd.compute_energy().real
        # except AssertionError:
        #    res_dict["ccd"] = np.nan

        # try:
        #    ccsd = CCSD(system, verbose=True)
        #    ccsd.compute_ground_state(t_kwargs=dict(tol=1e-3), l_kwargs=dict(tol=1e-3))
        #    res_dict["ccsd"] = ccsd.compute_energy().real
        # except AssertionError:
        #    res_dict["ccsd"] = np.nan

        try:
            oaccd = OACCD(system, verbose=True)
            oaccd.compute_ground_state(
                num_vecs=15, tol=1e-2, termination_tol=1e-2
            )
            res_dict["noccd"] = oaccd.compute_energy().real
        except AssertionError:
            res_dict["noccd"] = np.nan

        df = df.append(pd.Series(res_dict), ignore_index=True)


print(df)

path = os.path.join(sys.path[0], "dat")
filename = "gs_df.pkl"

df.to_pickle(os.path.join(path, filename))
