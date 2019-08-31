import numpy as np
import scipy.signal

import pint

from tdqd_tools.io_data import return_time_data

solvers = ["tdhf", "oatdccd"]
basis_sets = ["sto-3g", "6-31gss"]
molecules = [
    "h2",
    # "lih",
    # "co",
]
spins = ["both", "up"]

ureg = pint.UnitRegistry()

for molecule in molecules:
    for basis in basis_sets:
        for spin in spins:
            print(f"{molecule} ({basis}) spin: {spin}")
            for solver in solvers:
                filename = f"fft_{molecule}_{basis}_{solver}_{spin}_real.dat"

                data = return_time_data(filename)
                time, spec = data[:, 0], data[:, 1]

                peak_indices = scipy.signal.find_peaks(spec, height=1e-1)

                peaks = list(time[peak_indices[0]])

                for i, peak in enumerate(peaks):
                    energy_au = peak
                    energy_ev = (peak * ureg.hartree).to(ureg.eV).magnitude

                    tab_entry = rf"& ${i + 1}$ & ${energy_au:.4f}$ & ${energy_ev:.4f}$ \\"
                    if i == 0:
                        tab_entry = solver.upper() + " " + tab_entry

                    print(tab_entry)

            print("\n\n")
