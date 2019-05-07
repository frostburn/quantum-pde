from pylab import *

from flow import schrodinger_flow_2D
from util import normalize_2D, advance_pde
from lattice import make_lattice_2D, make_border_wall_2D, make_periodic_2D


def static_gaussian(resolution):
    x, y, dx, screen = make_lattice_2D(resolution, 5, 0.5)
    wall = make_border_wall_2D(resolution, 5, 0.5, weight=1000)
    potential = wall

    psi_ = exp(-10*(x**2 + y**2)) + 0j
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 0.2

    return dx, screen, psi, potential, episode_length


def double_slit(resolution):
    x, y, dx, screen = make_lattice_2D(resolution, 10, 5)
    wall = make_border_wall_2D(resolution, 10, 5, weight=100)
    potential = wall + exp(-(3*(x+1))**4) * (1 - exp(-(5*(y-1))**4) - exp(-(5*(y+1))**4)) * 2000

    psi_ = exp(-10*(x+4)**2 - 10*y**2 + 10j*x)
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 0.55

    return dx, screen, psi, potential, episode_length


EPISODES = {
    "static_gaussian": static_gaussian,
    "double_slit": double_slit,
}
