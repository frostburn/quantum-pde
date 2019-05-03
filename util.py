# coding: utf-8
import numpy as np

def advance_pde(t, psi, potential, dt, dx, flow, stencil=1, dimensions=2):
    """
    Advances a partial differential equation defined by `flow`
    by the amount `dt` using the Runge-Kutta method.

    :param t: Current time for time-dependent flows
    :param psi: Current solutions to the PDE
    :param potential: Static potential. A parameter of `flow`
    :param dt: Small span of time. The PDE solution is advanced to time `t+dt`
    :param dx: Lattice constant of `psi`
    :param flow: Function of (t, psi, potential, dx)
    :param stencil: How much does `flow` eat into the lattice. The result from this method eats 4 times as much.
    :param dimensions: How many dimensions does `psi` have.
    """
    s = stencil
    s1 = (slice(s, -s),)*dimensions
    s2 = (slice(2*s, -2*s),)*dimensions
    s3 = (slice(3*s, -3*s),)*dimensions
    s4 = (slice(4*s, -4*s),)*dimensions

    k1 = flow(t, psi, potential, dx)
    k2 = flow(t + 0.5*dt, psi[s1] + (0.5*dt) * k1, potential[s1], dx)
    k3 = flow(t + 0.5*dt, psi[s2] + (0.5*dt) * k2, potential[s2], dx)
    k4 = flow(t + dt, psi[s3] + dt * k3, potential[s3], dx)

    # We're effectively doing
    # return psi[s4] + (k1[s3] + 2*k2[s2] + 2*k3[s1] + k4) * dt / 6.0
    # but like this we avoid creating temporary arrays.

    accum = k4
    accum += k1[s3]
    accum += k2[s2]
    accum += k2[s2]
    accum += k3[s1]
    accum += k3[s1]

    return psi[s4] + accum*(dt/6.0)


def laplacian_1D(psi, dx):
    """
    Calculate a discrete approximation to the laplacian on a
    one dimensional lattice `psi` with lattice constant `dx`.
    Has a stencil of one on each side so the result is a smaller lattice.
    """
    # Doing the laplacian like this avoids temporary arrays.
    result = (-2)*psi[1:-1]
    result += psi[:-2]
    result += psi[2:]
    return result / (dx*dx)


def laplacian_2D(psi, dx):
    """
    Calculate a discrete approximation to the laplacian on a
    two dimensional lattice `psi` with lattice constant `dx`.
    Has a stencil of one on each side so the result is a smaller lattice.
    """
    # Doing the laplacian like this avoids temporary arrays.
    result = (-4)*psi[1:-1, 1:-1]
    result += psi[2:, 1:-1]
    result += psi[:-2, 1:-1]
    result += psi[1:-1, 2:]
    result += psi[1:-1, :-2]
    return result / (dx*dx)


def normalize_1D(psi, dx):
    """
    Normalize a probability amplitude to unity
    """
    total_probability = (abs(psi)**2).sum()*dx
    return psi / np.sqrt(total_probability)


def normalize_2D(psi, dx):
    """
    Normalize a probability amplitude to unity
    """
    total_probability = (abs(psi)**2).sum()*dx*dx
    return psi / np.sqrt(total_probability)
