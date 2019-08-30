import numpy as np
import scipy.signal

import pint

from tdqd_tools.io_data import return_time_data

directions = ["x", "z"]
solvers = ["tdhf", "tdccsd", "oatdccd"]

ureg = pint.UnitRegistry()

for direction in directions:
    print(f"Direction: {direction}")
    for solver in solvers:
        filename = f"fft_dipole_{direction}_{solver}_real_new.dat"

        data = return_time_data(filename)
        time, spec = data[:, 0], data[:, 1]

        peak_indices = scipy.signal.find_peaks(spec, height=1e-6)

        peaks = list(time[peak_indices[0]])

        for i, peak in enumerate(peaks):
            energy_au = peak
            energy_ev = (peak * ureg.hartree).to(ureg.eV).magnitude

            tab_entry = (
                rf"& ${i + 1}$ & ${energy_au:.4f}$ & ${energy_ev:.4f}$ \\"
            )
            if i == 0:
                tab_entry = solver.upper() + " " + tab_entry

            print(tab_entry)

    print("\n\n")
