from pylab import *


def static_gaussian(num_particles):
    positions = randn(num_particles, 2) * 0.5
    velocities = randn(num_particles, 2) * 0.1

    return {
        "positions": positions,
        "velocities": velocities,
        "episode_length": 20.0,
    }


def superposition(num_particles):
    half = num_particles // 2
    positions = randn(half, 2) * 0.1
    shift = positions * 0
    shift[:, 0] = -6
    shift[:, 1] = 1.5
    positions += shift
    positions = concatenate((positions, randn(num_particles - half, 2) * 0.5))
    shift = positions * 0
    shift[:, 0] = 3
    shift[:, 1] = -0.5
    positions += shift
    velocities = randn(num_particles, 2) * 0.1

    return {
        "positions": positions,
        "velocities": velocities,
        "episode_length": 20.0,
    }


def colliding_superposition(num_particles):
    half = num_particles // 2
    positions = randn(half, 2) * 0.1
    shift = positions * 0
    shift[:, 0] = -6
    shift[:, 1] = 1.5
    positions += shift
    positions = concatenate((positions, randn(num_particles - half, 2) * 0.5))
    shift = positions * 0
    shift[:, 0] = 3
    shift[:, 1] = -0.5
    positions += shift
    velocities = randn(half, 2) * 0.1
    push = velocities * 0
    push[:, 0] = 1
    velocities += push
    velocities = concatenate((velocities, randn(num_particles - half, 2) * 0.1))
    push = velocities * 0
    push[:, 0] = -0.5
    velocities += push

    return {
        "positions": positions,
        "velocities": velocities,
        "episode_length": 20.0,
    }


def tunneling(num_particles, velocity=0.75):
    def potential(x, y):
        # return exp(-x**4) / 4.0 - 0.001 * y**4 / 4.0
        return exp(-x**4) * 0.25

    def force(pos):
        x = pos[:, 0]
        y = pos[:, 1]
        ax = x**3 * exp(-x**4)
        ay = -0.001*y**3
        return array([ax, ay]).T

    positions = randn(num_particles, 2) * 0.5
    shift = positions * 0
    shift[:, 0] = -4
    positions += shift
    velocities = randn(num_particles, 2) * 0.1
    push = velocities * 0
    push[:, 0] = velocity
    velocities += push

    return {
        "positions": positions,
        "velocities": velocities,
        "force": force,
        "potential": potential,
        "episode_length": 30.0,
    }


def convex_mirror(num_particles):
    def potential(x, y):
        curve = (x - 5.5 + 0.07*y**2)
        return exp(-curve**4)

    def force(pos):
        x = pos[:, 0]
        y = pos[:, 1]
        curve = (x - 5.5 + 0.07*y**2)
        potential = exp(-curve**4)
        ax = 4 * (curve) ** 3 * potential
        ay = 4 * 0.07 * 2 * y * (curve) ** 3 * potential
        return array([ax, ay]).T

    positions = randn(num_particles, 2) * 0.5
    shift = positions * 0
    shift[:, 0] = -4
    positions += shift
    velocities = randn(num_particles, 2) * 0.1
    push = velocities * 0
    push[:, 0] = 0.75
    velocities += push

    return {
        "positions": positions,
        "velocities": velocities,
        "force": force,
        "potential": potential,
        "episode_length": 40.0,
    }


def double_slit(num_particles):
    def potential(x, y):
        wall = exp(-(4*(x+1))**4)
        holes = 1 - exp(-(4*(y+1))**4) - exp(-(4*(y-1))**4)
        return wall * holes

    def force(pos):
        x = pos[:, 0]
        y = pos[:, 1]
        wall = exp(-(4*(x+1))**4)
        holes = 1 - exp(-(4*(y+1))**4) - exp(-(4*(y-1))**4)
        ax = 1024 * (x+1)**3 * wall * holes
        ay = -((y+1)**3 * exp(-(4*(y+1))**4) + (y-1)**3 * exp(-(4*(y-1))**4)) * wall * 1024
        return array([ax, ay]).T

    positions = randn(num_particles, 2) * 0.5
    shift = positions * 0
    shift[:, 0] = -5
    positions += shift
    velocities = randn(num_particles, 2) * 0.1
    push = velocities * 0
    push[:, 0] = 0.3
    velocities += push

    return {
        "positions": positions,
        "velocities": velocities,
        "potential": potential,
        "force": force,
        "episode_length": 70.0,
    }


def square_measurement(num_particles, inverted=False):
    def in_the_box(pos):
        x = pos[:, 0]
        y = pos[:, 1]
        return logical_and(logical_and(logical_and(0.3 < x, x < 1.8), -0.1 > y), y > -1.6)

    if inverted:
        measurement = lambda pos: ~in_the_box(pos)
    else:
        measurement = in_the_box

    positions = randn(num_particles, 2) * 0.5
    velocities = randn(num_particles, 2) * 0.1

    return {
        "positions": positions,
        "velocities": velocities,
        "measurements": {1.0: measurement},
        "episode_length": 10.0,
    }


EPISODES = {
    "square_measurement": square_measurement,
    "square_measurement_inverted": lambda n: square_measurement(n, inverted=True),
    "superposition": superposition,
    "double_slit": double_slit,
    "convex_mirror": convex_mirror,
    "tunneling": tunneling,
    "tunneling_slow": lambda n: tunneling(n, 0.3),
    "colliding_superposition": colliding_superposition,
    "static_gaussian": static_gaussian,
}
