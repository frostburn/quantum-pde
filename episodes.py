from pylab import *

from flow import schrodinger_flow_2D
from util import normalize_2D, advance_pde
from lattice import make_lattice_2D, make_border_wall_2D, make_periodic_2D


def static_gaussian(resolution):
    x, y, dx, screen = make_lattice_2D(resolution, 5, 5)
    wall = make_border_wall_2D(resolution, 5, 5, weight=100)
    potential = wall

    psi_ = exp(-15*(x**2 + y**2)) + 0j
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 0.2

    return dx, screen, psi, potential, episode_length


def gaussian_superposition(resolution):
    x, y, dx, screen = make_lattice_2D(resolution, 5, 5)
    potential = 0*x

    psi_ = exp(-60*((x+1.5)**2 + (y+0.2)**2)) + exp(-5*((x-1.5)**2 + (y-0.2)**2)) + 0j
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 0.15

    return dx, screen, psi, potential, episode_length


def moving_gaussian(resolution):
    x, y, dx, screen = make_lattice_2D(resolution, 5, 2)
    potential = 0*x

    psi_ = exp(-15*((x+1.5)**2 + y**2) + 30j*x) + 0j
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 0.1

    return dx, screen, psi, potential, episode_length


def colliding_gaussians(resolution):
    x, y, dx, screen = make_lattice_2D(resolution, 5, 5)
    potential = 0*x

    psi_ = exp(-15*((x+1.5)**2 + (y+0.2)**2) + 31j*x) + exp(-17*((x-1.5)**2 + (y-0.2)**2) - 28j*x)
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 0.15

    return dx, screen, psi, potential, episode_length


def convex_mirror(resolution):
    x, y, dx, screen = make_lattice_2D(resolution, 5, 4)
    potential = exp(-(x-3.3+0.12*y*y)**4) * 7000

    psi_ = exp(-30*((x+1.5)**2 + y**2) + 35j*x)
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 0.18

    return dx, screen, psi, potential, episode_length


def tunneling(resolution, weight=200, omega=14.75):
    x, y, dx, screen = make_lattice_2D(resolution, 7, extra_width=4, extra_height=0)
    potential = exp(-(2*x)**6) * weight
    potential = maximum(potential, (1 - exp(-(0.35*y)**6)) * 1000)

    psi_ = exp(-1.1*((x+2.5)**2 + y**2) + omega*1j*x)
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 0.4

    return dx, screen, psi, potential, episode_length


def double_slit(resolution):
    x, y, dx, screen = make_lattice_2D(resolution, 10, 5)
    wall = make_border_wall_2D(resolution, 10, 5, weight=100)
    potential = wall + exp(-(3*(x+1))**4) * (1 - exp(-(5*(y-1))**4) - exp(-(5*(y+1))**4)) * 2000

    psi_ = exp(-1*(x+4)**2 - 10*y**2 + 15j*x)
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 0.55

    return dx, screen, psi, potential, episode_length


EPISODES = {
    "static_gaussian": static_gaussian,
    "gaussian_superposition": gaussian_superposition,
    "moving_gaussian": moving_gaussian,
    "colliding_gaussians": colliding_gaussians,
    "convex_mirror": convex_mirror,
    "double_slit": double_slit,
    "tunneling": tunneling,
}
