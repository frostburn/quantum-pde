from pylab import *


def superposition(num_particles):
    half = num_particles // 2
    positions = randn(half, 2) * 0.5
    shift = positions * 0
    shift[:, 0] = -4
    shift[:, 1] = 1
    positions += shift
    positions = concatenate((positions, randn(num_particles - half, 2)))
    shift = positions * 0
    shift[:, 0] = 2
    shift[:, 1] = -0.5
    positions += shift
    velocities = randn(num_particles, 2) * 0.1

    return {
        "positions": positions,
        "velocities": velocities,
    }


def tunneling(num_particles):
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
    push[:, 0] = 0.75
    velocities += push

    return positions, velocities, force


def convex_mirror(num_particles):
    # potential = exp(-(x-3.3+0.12*y*y)**4) * 7000

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

    return positions, velocities, force


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
    }
