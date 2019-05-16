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


def tunneling(resolution, weight=200, omega=14.75, extra_width=4):
    x, y, dx, screen = make_lattice_2D(resolution, 7, extra_width=extra_width, extra_height=0)
    potential = exp(-(2*x)**6) * weight
    potential = maximum(potential, (1 - exp(-(0.35*y)**6)) * 1000)

    psi_ = exp(-1.1*((x+2.5)**2 + y**2) + omega*1j*x)
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 0.4

    return dx, screen, psi, potential, episode_length


def slit_base(resolution):
    x, y, dx, screen = make_lattice_2D(resolution, 10, 6)
    potential = exp(-(8*(x+1))**4) * 2000
    psi_ = exp(-10*(x+4)**2 - 10*y**2 + 15j*x)
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)
    episode_length = 0.5

    return x, y, dx, screen, psi, potential, episode_length

def single_slit(resolution):
    x, y, dx, screen, psi, potential, episode_length = slit_base(resolution)
    potential *= 1 - exp(-(5*(y+1))**4)
    return dx, screen, psi, potential, episode_length


def double_slit(resolution):
    x, y, dx, screen, psi, potential, episode_length = slit_base(resolution)
    potential *= 1 - exp(-(5*(y+1))**4) - exp(-(5*(y-1))**4)
    return dx, screen, psi, potential, episode_length


def double_slit_measured(resolution):
    x, y, dx, screen, psi, potential, episode_length = slit_base(resolution)
    potential *= 1 - exp(-(5*(y+1))**4) - exp(-(5*(y-1))**4)

    mask = 1 - exp(-((4*(x+0.9))**8 + (4*(y-1))**8))
    measurements = {}

    for i in range(100):
        measurements[episode_length * i / 100.0] = {"mask": mask, "forced": True}

    return {
        "dx": dx,
        "screen": screen,
        "psi": psi,
        "potential": potential,
        "episode_length": episode_length,
        "measurements": measurements
    }


def box_with_stuff(resolution):
    x, y, dx, screen = make_lattice_2D(resolution, 10, 0.5)
    wall = make_border_wall_2D(resolution, 10, 0.5, weight=1000)
    potential = wall
    potential += exp(-(x-0)**2 - (y-1.8)**2) * 60
    potential += exp(-(x+y*0.7)**8 - (x*0.7-y-2)**8) * 1000

    psi_ = exp(-2*(x+2.5)**2 - 2*y**2 + 5j*x)
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 3.0

    return dx, screen, psi, potential, episode_length


def gaussian_measured(resolution, inverted=True):
    x, y, dx, screen = make_lattice_2D(resolution, 5, 5)
    wall = make_border_wall_2D(resolution, 5, 5, weight=100)
    potential = wall

    psi_ = exp(-50*(x**2 + y**2)) + 0j
    psi = psi_ * 0
    psi[4:-4, 4:-4] = psi_[4:-4, 4:-4]
    psi = normalize_2D(psi, dx)

    episode_length = 0.1

    mask = exp(-10000*((x-0.2)**20 + (y+0.3)**20))
    if inverted:
        mask = 1 - mask

    return {
        "dx": dx,
        "screen": screen,
        "psi": psi,
        "potential": potential,
        "episode_length": episode_length,
        "measurements": {
            0.02: {"mask": mask, "forced": True},
        },
    }


EPISODES = {
    "static_gaussian": static_gaussian,
    "gaussian_superposition": gaussian_superposition,
    "moving_gaussian": moving_gaussian,
    "colliding_gaussians": colliding_gaussians,
    "convex_mirror": convex_mirror,
    "single_slit": single_slit,
    "double_slit": double_slit,
    "box_with_stuff": box_with_stuff,
    "tunneling": tunneling,
    "tunneling_slow": lambda r: tunneling(r, omega=9.0),
    "tunneling_fast": lambda r: tunneling(r, omega=18.0, extra_width=6),
    "gaussian_measured": gaussian_measured,
    "gaussian_measured_inverted": lambda r: gaussian_measured(r, inverted=False),
    "double_slit_measured": double_slit_measured,
}
