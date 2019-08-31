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


for I in range(len(cisd_no.energies)):
    if cisd_no.energies[I] - cisd_no.energies[0] > 1.8:
        break

    print(f"I = {I}")
    print("Spin-independent:")
    print(cisd_no.allowed_dipole_transition(I, 0))
    print(cisd_no.energies[I] - cisd_no.energies[0])

    print("---")
    print("Spin-dependent")
    print(cisd_yes.allowed_dipole_transition(I, 0))
    print(cisd_yes.energies[I] - cisd_yes.energies[0])
    print("=" * 100)
