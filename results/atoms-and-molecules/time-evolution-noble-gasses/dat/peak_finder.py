import numpy as np
import scipy.signal

import pint

from tdqd_tools.io_data import return_time_data

solvers = ["oatdccd"]
basis_sets = ["aug-ccpvdz"]

ureg = pint.UnitRegistry()

for basis in basis_sets:
    for solver in solvers:
        filename = f"fft_ne_{basis}_{solver}_new_real.dat"

        data = return_time_data(filename)
        time, spec = data[:, 0], data[:, 1]

        peak_indices = scipy.signal.find_peaks(spec, height=0.01)

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
