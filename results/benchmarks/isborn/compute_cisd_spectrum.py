import pint
import numpy as np
import matplotlib.pyplot as plt

from isborn_system import get_h2_system

from configuration_interaction import CISD, CIS

h2_no = get_h2_system(spin_dependent=False, basis="6-31gss")
h2_yes = get_h2_system(spin_dependent=True, basis="6-31gss")

cisd_no = CISD(h2_no, verbose=True)
cisd_no.compute_ground_state()

cisd_yes = CISD(h2_yes, verbose=True)
cisd_yes.compute_ground_state()

ureg = pint.UnitRegistry()


for J in range(len(cisd_no.energies)):
    if cisd_no.energies[J] - cisd_no.energies[0] > 1.0:
        break

    allowed_dipole = cisd_no.allowed_dipole_transition(0, J)
    singlet_allowed = "no"

    if np.abs(allowed_dipole[2]) > 1e-12:
        singlet_allowed = "yes"

    allowed_dipole = cisd_yes.allowed_dipole_transition(0, J)
    triplet_allowed = "no"

    if np.abs(allowed_dipole[2]) > 1e-12:
        triplet_allowed = "yes"

    if triplet_allowed == "no" and singlet_allowed == "no":
        continue

    energy_au = (cisd_no.energies[J] - cisd_no.energies[0]) * ureg.hartree
    energy_ev = energy_au.to(ureg.eV).magnitude
    energy_au = energy_au.magnitude

    assert abs(energy_au - cisd_yes.energies[J] + cisd_yes.energies[0]) < 1e-12

    tab_entry = rf"$0 \to {J}$ & {singlet_allowed} & ${energy_au:.4f}$ & ${energy_ev:.4f}$ \\"
    print(tab_entry)

    #print(f"J = {J}")
    #print("Spin-independent:")
    #print(cisd_no.allowed_dipole_transition(0, J))
    #print(cisd_no.energies[J] - cisd_no.energies[0])

    #print("---")
    #print("Spin-dependent")
    #print(cisd_yes.allowed_dipole_transition(0, J))
    #print(cisd_yes.energies[J] - cisd_yes.energies[0])
    #print("=" * 100)
