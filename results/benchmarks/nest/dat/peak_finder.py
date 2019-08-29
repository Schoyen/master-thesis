import numpy as np
import scipy.signal

from tdqd_tools.io_data import return_time_data

data = return_time_data("fft_dipole_x_tdhf_real_new.dat")
time, spec = data[:, 0], data[:, 1]

peak_indices = scipy.signal.find_peaks(spec)

print(time[peak_indices[0]])
