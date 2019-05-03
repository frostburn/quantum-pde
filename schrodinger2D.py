# coding: utf-8
from __future__ import division
from matplotlib.animation import FuncAnimation
from pylab import *

from flow import schrodinger_flow_2D
from util import normalize_2D, advance_pde
from lattice import make_lattice_2D, make_border_wall_2D

resolution = "80p"

x, y, dx, screen = make_lattice_2D(resolution, 10, 5)
potential = make_border_wall_2D(resolution, 10, 5, weight=100)
potential *= rand(*potential.shape)
potential = exp(-(3*(x+1))**4) * (1 - exp(-(5*(y-1))**4) - exp(-(5*(y+1))**4)) * 2000

t = 0
dt = 0.004 * dx

psi_ = exp(-10*(x+4)**2 - 10*y**2 + 10j*x)
psi = psi_ * 0
psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
psi = normalize_2D(psi, dx)

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
        t += dt
    psi = normalize_2D(psi, dx)
    prob = abs(psi)**2
    impsi.set_data(prob[screen])
    return impsi,

ani = FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True, repeat=True, interval=1)
show()
