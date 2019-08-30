import numpy as np
import pint

from lih_nest_system import get_lih_system

from configuration_interaction import CISD

ureg = pint.UnitRegistry()

cisd = CISD(get_lih_system(), verbose=True)
cisd.compute_ground_state()
print(cisd.energies[:10])
print(cisd.energies[:40] - cisd.energies[0])

directions = ["x", "y", "z"]

for J in range(1, 41):

    allowed_dipole = cisd.allowed_dipole_transition(0, J)
    allowed_direction = np.abs(allowed_dipole) > 1e-12

    if not np.any(allowed_direction):
        continue

    direction = ""

    if allowed_direction[0]:
        direction = "x"
        if allowed_direction[1]:
            direction += "$, $y"
    else:
        direction = "z"

    delta_E_au = (cisd.energies[J] - cisd.energies[0]) * ureg.hartree
    delta_E_ev = delta_E_au.to(ureg.eV).magnitude

    tab_entry = rf"$0 \to {J}$ & ${direction}$ & ${delta_E_au.magnitude:.4f}$ & ${delta_E_ev:.4f}$ \\"
    print(tab_entry)
