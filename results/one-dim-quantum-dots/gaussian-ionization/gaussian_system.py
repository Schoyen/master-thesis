import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from quantum_systems import ODQD
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import (
    GaussianPotential,
)
from quantum_systems.time_evolution_operators import LaserField

from coupled_cluster.ccd import OATDCCD
from hartree_fock import HartreeFock


n = 2
l = 20

weight = 4
center = 0
deviation = 1

grid_length = 30
num_grid_points = 601

odqd = ODQD(n, l, grid_length, num_grid_points)
odqd.setup_system(potential=GaussianPotential(weight, center, deviation, np))

omega = 1
E = 10

laser_pulse = lambda t: E * np.sin(omega * t)
odqd.set_time_evolution_operator(LaserField(laser_pulse))

plt.plot(odqd.grid, odqd.potential(odqd.grid))

for i in range(l // 2):
    plt.plot(
        odqd.grid,
        odqd.eigen_energies[i] + np.abs(odqd.spf[2 * i]) ** 2,
        label=rf"i = {i}",
    )

plt.legend(loc="best")
plt.show()

hf = HartreeFock(odqd, verbose=True)
hf.compute_ground_state(tol=1e-10)
odqd.change_basis(hf.C)


oatdccd = OATDCCD(odqd, verbose=True)
oatdccd.compute_ground_state(tol=1e-10, termination_tol=1e-10)

t_start = 0
t_end = 10
dt = 1e-2

num_points = int((t_end - t_start) / dt + 1)
time_points = np.linspace(t_start, t_end, num_points)

fig = plt.figure()
ax = plt.axes(xlim=(-grid_length, grid_length), ylim=(0, n))
line, = ax.plot([], [])


oatdccd.set_initial_conditions()
gen = oatdccd.solve(time_points)


def init():
    line.set_data([], [])
    return (line,)


def animate(i):
    gen.__next__()
    line.set_data(odqd.grid, oatdccd.compute_particle_density())
    return (line,)


anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=20, interval=20, blit=True
)

plt.show()
