import numpy as np
import scipy.signal

from tdqd_tools.io_data import return_time_data

directions = ["x", "z"]
solvers = ["tdhf", "tdccsd", "oatdccd"]

for direction in directions:
    for solver in solvers:
        filename = f"fft_dipole_{direction}_{solver}_real_new.dat"

        data = return_time_data(filename)
        time, spec = data[:, 0], data[:, 1]

        peak_indices = scipy.signal.find_peaks(spec)
        print(filename)
        print(time[peak_indices[0]])
