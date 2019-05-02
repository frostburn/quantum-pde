from matplotlib.animation import FuncAnimation
from pylab import *

from util import advance_pde, normalize
from flow import schrodinger_flow_1D

x = linspace(-15, 15, 512)

dx = x[1] - x[0]
dt = 0.004*dx

psi = exp(-x*x -2j*x)

border = 13.5
potential = where(x < -border, (x+border)**2, where(x > border, (x-border)**2, 0))*1000

fig, ax = subplots()
lnpsi, = plot(x, abs(psi)**2)

def init():
    ax.set_ylim(0, 1)
    return lnpsi,

def update(frame):
    global psi
    for j in range(10):
        patch = advance_pde(0, psi, potential, dt, dx, schrodinger_flow_1D, dimensions=1)
        psi[4:-4] = patch
    psi = normalize(psi, dx)
    if frame == 0:
        print("Energy = {}".format((dx*abs(schrodinger_flow_1D(0, psi, potential, dx))**2).sum()))
    prob = abs(psi)**2
    lnpsi.set_ydata(prob)
    return lnpsi,

ani = FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True, repeat=True, interval=1)
show()
