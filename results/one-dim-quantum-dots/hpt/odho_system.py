import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from hpt_system import get_hpt_system

from configuration_interaction import TDCISD
from coupled_cluster.ccd import OATDCCD
from hartree_fock import HartreeFock


n = 2
l = 20

omega = 1

grid_length = 10
num_grid_points = 401

from quantum_systems import ODQD
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import (
    HOPotential,
    DWPotential,
)
from quantum_systems.time_evolution_operators import LaserField

odqd = ODQD(n, l, grid_length, num_grid_points)
odqd.setup_system(potential=DWPotential(omega=omega, l=2))


laser_omega = 1
E = 1

laser_pulse = lambda t: E * np.sin(laser_omega * t)
laser_envelope = lambda t, T: np.sin(np.pi * t / T) ** 2 if t <= T else 0
laser = lambda t: laser_envelope(t, 5) * laser_pulse(t)
odqd.set_time_evolution_operator(LaserField(laser))

# odqd = get_hpt_system(n=1)


plt.plot(odqd.grid, odqd.potential(odqd.grid))

for i in range(odqd.l // 2):
    plt.plot(
        odqd.grid,
        odqd.eigen_energies[i] + np.abs(odqd.spf[2 * i]) ** 2,
        label=rf"i = {i}",
    )

plt.legend(loc="best")
plt.show()


tdcisd = TDCISD(odqd, verbose=True)
tdcisd.compute_ground_state()
tdcisd.set_initial_conditions()

background = odqd.potential(odqd.grid)

plt.plot(odqd.grid, background)
plt.fill_between(odqd.grid, background, alpha=0.3)
plt.plot(odqd.grid, tdcisd.compute_particle_density().real)
plt.ylim(0, 1)
plt.show()

t_start = 0
t_end = 10
dt = 1e-2

num_points = int((t_end - t_start) / dt + 1)
time_points = np.linspace(t_start, t_end, num_points)


fig = plt.figure()
ax = plt.axes(xlim=(odqd.grid[0], odqd.grid[-1]), ylim=(0, odqd.n))
line, = ax.plot([], [])
title = ax.text(0.5, 0.85, "")

gen = enumerate(tdcisd.solve(time_points))


def init():
    time_step = 0
    title.set_text(f"t = 0")
    line.set_data([], [])
    return (line, title)


def animate(i):
    time_step, amp = next(gen)
    title.set_text(f"t = {time_step * dt:.2f}")
    line.set_data(odqd.grid, tdcisd.compute_particle_density().real)
    return (line, title)


plt.plot(odqd.grid, background)
plt.fill_between(odqd.grid, background, alpha=0.3)
anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=20, interval=20, blit=True
)

plt.show()
