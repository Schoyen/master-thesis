import numpy as np

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import LaserField

from noble_laser import NobleLaser


def get_system(atom, basis, num_cycles):
    system = construct_pyscf_system_rhf(atom, basis)

    polarization_vector = np.zeros(3)
    # Negative polarization as we look at electrons
    polarization_vector[2] = -1

    system.set_time_evolution_operator(
        LaserField(
            NobleLaser(num_cycles), polarization_vector=polarization_vector
        )
    )

    return system


if __name__ == "__main__":
    import pint
    from coupled_cluster.ccd import OACCD
    from coupled_cluster.ccsd import CoupledClusterSinglesDoubles

    ureg = pint.UnitRegistry()

    atoms = ["ne", "ar"]
    basis_sets = ["ccpvdz", "aug-ccpvdz"]
    basis_names = {"ccpvdz": "cc-pVDZ", "aug-ccpvdz": "aug-cc-pVDZ"}
    atom_names = {"ne": r"\ch{Ne}", "ar": r"\ch{Ar}"}

    for atom in atoms:
        for basis in basis_sets:
            system = get_system(atom, basis, 1)

            oaccd = OACCD(system)
            oaccd.compute_ground_state(tol=1e-6, termination_tol=1e-6)
            oaccd_au = oaccd.compute_energy()
            oaccd_ev = (oaccd_au * ureg.hartree).to(ureg.eV).magnitude

            tab_entry = rf"{atom_names[atom]} & {basis_names[basis]} & OATDCCD & ${oaccd_au:.4f}$ & ${oaccd_ev:.4f}$ \\"
            print(tab_entry)

            system = get_system(atom, basis, 1)
            ccsd = CoupledClusterSinglesDoubles(system)
            ccsd.compute_ground_state(
                t_kwargs=dict(tol=1e-6), l_kwargs=dict(tol=1e-6)
            )
            ccsd_au = ccsd.compute_energy()
            ccsd_ev = (ccsd_au * ureg.hartree).to(ureg.eV).magnitude

            tab_entry = rf"{atom_names[atom]} & {basis_names[basis]} & TDCCSD & ${ccsd_au:.4f}$ & ${ccsd_ev:.4f}$ \\"
            print(tab_entry)
