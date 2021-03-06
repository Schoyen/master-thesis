import numpy as np

from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles as CCSD


def get_ar2_system(bond, basis="ccpvdz"):
    mol = f"ar 0.0 0.0 {-bond / 2}; ar 0.0 0.0 {bond / 2}"
    mol = f"h 0.0 0.0 {-bond /2}; h 0.0 0.0 {bond / 2}"

    return construct_pyscf_system_rhf(mol, basis=basis)


bonds = np.linspace(0.5, 5, 101)
energies = np.zeros(len(bonds), dtype=np.complex128)
energies_no_nuc = np.zeros_like(energies)

for i, bond in enumerate(bonds):
    ccsd = CCSD(get_ar2_system(bond), verbose=True)
    ccsd.compute_ground_state()

    energies[i] = ccsd.compute_energy() + ccsd.system.nuclear_repulsion_energy
    energies_no_nuc[i] = ccsd.compute_energy()

energies = energies.real
energies_no_nuc = energies_no_nuc.real
np.savetxt(
    "lennard-jones-ccsd-ish.dat",
    np.c_[bonds[:, np.newaxis], energies[:, np.newaxis]],
)
np.savetxt(
    "lennard-jones-ccsd-no-nuc.dat",
    np.c_[bonds[:, np.newaxis], energies_no_nuc[:, np.newaxis]],
)
