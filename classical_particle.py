from __future__ import division

from matplotlib.animation import FuncAnimation

from pylab import *
import numpy.random

from lattice import RESOLUTIONS
from classical_episodes import tunneling, convex_mirror, square_measurement, superposition

width, height = RESOLUTIONS["80p"]

num_particles = 2 * width * height

scale = 0.1

episode = superposition(num_particles)
if isinstance(episode, tuple):
    positions, velocities, force = episode
else:
    locals().update(episode)

locals().setdefault("force", lambda pos: pos*0)
locals().setdefault("measurements", {})

t = 0.0
dt = 0.05

counts = zeros((height, width), dtype=complex)

def step():
    global t, positions, velocities, phases, counts
    for measurement_t in list(measurements.keys()):
        if t >= measurement_t:
            print(t)
            condition = measurements.pop(measurement_t)
            keepers = condition(positions)
            positions[~keepers] += 10
            velocities[~keepers] = 0
    counts *= 0
    for pos in positions:
        i = int(pos[0] * height * scale + width * 0.5)
        j = int(pos[1] * height * scale + height * 0.5)
        if i >= 0 and i < width and j >= 0 and j < height:
            counts[j, i] += 1
    velocities += dt * force(positions)
    positions += dt * velocities
    t += dt

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
