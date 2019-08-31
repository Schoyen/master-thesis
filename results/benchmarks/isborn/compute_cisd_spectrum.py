import numpy as np

from isborn_system import get_h2_system

from configuration_interaction import CISD, CIS

h2_no = get_h2_system(spin_dependent=False, basis="sto-3g")
h2_yes = get_h2_system(spin_dependent=True, basis="sto-3g")

cisd_no = CISD(h2_no, verbose=True)
cisd_no.compute_ground_state()

cisd_yes = CISD(h2_yes, verbose=True)
cisd_yes.compute_ground_state()


for I in range(len(cisd_no.energies)):
    print(f"I = {I}")
    print(cisd_no.allowed_dipole_transition(I, 0))
    print(cisd_no.energies[I] - cisd_no.energies[0])

    print("---")
    print(cisd_yes.allowed_dipole_transition(I, 0))
    print(cisd_yes.energies[I] - cisd_yes.energies[0])
    print("=" * 100)
