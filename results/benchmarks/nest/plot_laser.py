import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from tdqd_tools.io_data import write_data
from lih_nest_system import NestLaser, get_time_points


laser = np.vectorize(NestLaser())
time_points = get_time_points()

time_points = time_points[time_points <= 50]


path = os.path.join(sys.path[0], "dat")
write_data(
    os.path.join(path, "nest_laser_new.dat"), time_points, -laser(time_points)
)
