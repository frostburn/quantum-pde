# coding: utf-8
from __future__ import division
import sys

from matplotlib.animation import FuncAnimation
from pylab import *

from flow import schrodinger_flow_2D
from util import normalize_2D, advance_pde
from lattice import make_lattice_2D, make_border_wall_2D, make_periodic_2D

resolution = "360p"

x, y, dx, screen = make_lattice_2D(resolution, 10, 5)
wall = make_border_wall_2D(resolution, 10, 5, weight=100)
# wall *= rand(*wall.shape)
potential = wall + exp(-(3*(x+1))**4) * (1 - exp(-(5*(y-1))**4) - exp(-(5*(y+1))**4)) * 2000
# zeno = exp(-(0.01*x)**4 -(0.02*y)**4)

t = 0
dt = 0.004 * dx


episode_length = 0.6
video_length = 60.0
fps = 60.0

num_frames_desired = video_length * fps
num_frames_offered = episode_length / dt

ticks_per_frame = int(np.ceil(num_frames_offered / num_frames_desired))

# TODO: Adjust dt to guarantee integer ticks / frame

# zeno = pow(zeno, 1/dt)

psi_ = exp(-10*(x+4)**2 - 10*y**2 + 10j*x)
psi = psi_ * 0
psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
# make_periodic_2D(psi)
psi = normalize_2D(psi, dx)

frame = 0
while t <= episode_length:
    if frame % 100 == 0:
        print("t = {}, Energy = {}".format(t, (dx*dx*abs(schrodinger_flow_2D(0, psi, potential, dx))**2).sum()))
    with open("./rendered/double_slit/frame{:05}.dat".format(frame), "wb") as f:
        save(f, psi[screen])
    for j in range(ticks_per_frame):
        patch = advance_pde(0, psi, potential, dt, dx, schrodinger_flow_2D, dimensions=2)
        psi[4:-4, 4:-4] = patch
        t += dt
    psi = normalize_2D(psi, dx)
    frame += 1

sys.exit()


fig, ax = subplots()
prob = abs(psi)**2
impsi = imshow(prob[screen], vmin=0, vmax=0.004*prob.max())

def init():
    return impsi,

def update(frame):
    global psi, t
    if frame == 0:
        print("t = {}, Energy = {}".format(t, (dx*dx*abs(schrodinger_flow_2D(0, psi, potential, dx))**2).sum()))
    for j in range(2):
        patch = advance_pde(0, psi, potential, dt, dx, schrodinger_flow_2D, dimensions=2)
        psi[4:-4, 4:-4] = patch
        # make_periodic_2D(psi)
        # psi *= zeno
        t += dt
    psi = normalize_2D(psi, dx)
    prob = abs(psi)**2
    impsi.set_data(prob[screen])
    return impsi,

ani = FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True, repeat=True, interval=1)
show()
