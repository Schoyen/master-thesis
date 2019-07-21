import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from tdho_system import get_tdho
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles
from hartree_fock import HartreeFock
from hartree_fock.mix import DIIS


def get_filename_stub(params):
    return f"ccsd_n={params['n']}_l={params['l']}_omega={params['omega']}"


def run_ccsd_tdho(params, filename_stub):
    tdho = get_tdho(**params)

    hf_converged = False

    for tol in [1e-8, 1e-6, 1e-4]:
        try:
            hf = HartreeFock(tdho, mixer=DIIS, verbose=True)
            hf.compute_ground_state(
                tol=tol, change_system_basis=True, num_vecs=5
            )
            hf_converged = True
            break
        except AssertionError:
            pass

    assert hf_converged, "HF did not converge"

    ccsd_converged = False

    for tol in [1e-8, 1e-6, 1e-4]:
        try:
            ccsd = CoupledClusterSinglesDoubles(tdho, verbose=True)
            ccsd.compute_ground_state(
                t_kwargs=dict(tol=1e-8), l_kwargs=dict(tol=1e-8)
            )
            ccsd_converged = True
            break
        except AssertionError:
            pass

    assert ccsd_converged, "CCSD did not converge"

    rho = ccsd.compute_particle_density()

    path = os.path.join(sys.path[0], "dat")
    filename = filename_stub + "_rho_real.dat"
    filename = os.path.join(path, filename)

    np.savetxt(
        filename,
        np.c_[
            tdho.T.ravel()[:, np.newaxis],
            tdho.R.ravel()[:, np.newaxis],
            rho.real.ravel()[:, np.newaxis],
        ],
    )

    fig = plt.figure()
    fig.add_subplot(1, 1, 1, polar=True)

    plt.contourf(tdho.T, tdho.R, rho.real)
    plt.savefig(os.path.join(path, filename_stub + "_rho_real.pdf"))
