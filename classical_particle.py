from __future__ import division

from matplotlib.animation import FuncAnimation

from pylab import *
import numpy.random

from lattice import RESOLUTIONS

width, height = RESOLUTIONS["160p"]

scale = 0.1


def force(pos):
    x = pos[:, 0]
    y = pos[:, 1]
    ax = x**3*exp(-(x)**4)
    ay = -0.001*y**3
    return array([ax, ay]).T

num_particles = 2 * width * height
positions = randn(num_particles, 2) * 0.5
shift = positions * 0
shift[:, 0] = -4
positions += shift
velocities = randn(num_particles, 2) * 0.1
push = velocities * 0
push[:, 0] = 0.75
velocities += push

dt = 0.1

counts = zeros((height, width), dtype=complex)

def step():
    global positions, velocities, phases, counts
    counts *= 0
    for pos in positions:
        i = int(pos[0] * height * scale + width * 0.5)
        j = int(pos[1] * height * scale + height * 0.5)
        if i >= 0 and i < width and j >= 0 and j < height:
            counts[j, i] += 1
    velocities += dt * force(positions)
    positions += dt * velocities

step()
fig, ax = subplots()
imc = imshow(abs(counts), vmin=0, vmax=20, cmap='gray')

def init():
    return imc,

def update(frame):
    step()
    imc.set_data(abs(counts))
    return imc,

ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True, repeat=True, interval=1)
show()
