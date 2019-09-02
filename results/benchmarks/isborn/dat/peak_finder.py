import numpy as np
import scipy.signal

import pint

from tdqd_tools.io_data import return_time_data

solvers = ["tdhf", "oatdccd"]
basis_sets = ["sto-3g", "6-31gss"]
molecules = [
    "h2",
    "lih",
    # "co",
]
spins = ["up", "both"]

ureg = pint.UnitRegistry()

for molecule in molecules:
    for basis in basis_sets:
        for spin in spins:
            print(f"{molecule} ({basis}) spin: {spin}")
            for solver in solvers:
                filename = f"fft_{molecule}_{basis}_{solver}_{spin}_real.dat"

                data = return_time_data(filename)
                time, spec = data[:, 0], data[:, 1]

                height = 5e-2

                if molecule == "lih":
                    if basis == "sto-3g":
                        if spin == "both":
                            height = 1e-3
                        else:
                            height = 1e-2
                    else:
                        height = 2e-2

                peak_indices = scipy.signal.find_peaks(spec, height=height)

                peaks = list(time[peak_indices[0]])

                for i, peak in enumerate(peaks):
                    energy_au = peak
                    energy_ev = (peak * ureg.hartree).to(ureg.eV).magnitude
                    singlet = "no" if spin == "up" else "yes"

                    tab_entry = rf"& {singlet} & ${energy_au:.4f}$ & ${energy_ev:.4f}$ \\"
                    if i == 0:
                        tab_entry = (
                            basis.upper()
                            + " & "
                            + solver.upper()
                            + " "
                            + tab_entry
                        )
                    else:
                        tab_entry = "& " + tab_entry

                    print(tab_entry)

            print("\n\n")
