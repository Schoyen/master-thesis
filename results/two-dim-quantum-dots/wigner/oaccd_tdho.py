import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from tdho_system import get_tdho
from coupled_cluster.ccd import OACCD
from hartree_fock import HartreeFock
from hartree_fock.mix import DIIS


def get_filename_stub(params):
    return f"n={params['n']}_l={params['l']}_omega={params['omega']}"


def run_oaccd_tdho(params, filename_stub, hf_tol=1e-7, oaccd_tol=1e-4):
    path = os.path.join(sys.path[0], "dat")

    tdho = get_tdho(**params)

    hf = HartreeFock(tdho, mixer=DIIS, verbose=True)
    hf.compute_ground_state(tol=hf_tol, change_system_basis=True, num_vecs=10)

    rho_hf = hf.compute_particle_density()

    filename = "hf_" + filename_stub + "_rho_real.dat"
    filename = os.path.join(path, filename)

    oaccd = OACCD(tdho, verbose=True)
    oaccd.compute_ground_state(
        tol=oaccd_tol, termination_tol=oaccd_tol, num_vecs=20
    )

    filename = "oaccd_" + filename_stub + "_energy.dat"
    filename = os.path.join(path, filename)

    with open(filename, "w") as f:
        f.write(f"HF energy: {hf.compute_energy()}\n")
        f.write(f"OACCD energy: {oaccd.compute_energy()}\n")

    rho = oaccd.compute_particle_density()

    filename = "oaccd_" + filename_stub + "_rho_real.dat"
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
    plt.savefig(os.path.join(path, "oaccd_" + filename_stub + "_rho_real.pdf"))
