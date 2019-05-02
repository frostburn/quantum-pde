# coding: utf-8
from util import laplacian_1D, laplacian_2D


def schrodinger_flow_1D(t, psi, potential, dx):
    """
    The flow part of the 2 dimensional Schrödinger equation.
    """
    return 1j * laplacian_1D(psi, dx) - 1j * potential[1:-1] * psi[1:-1]


def schrodinger_flow_2D(t, psi, potential, dx):
    """
    The flow part of the 2 dimensional Schrödinger equation.
    """
    return 1j * laplacian_2D(psi, dx) - 1j * potential[1:-1, 1:-1] * psi[1:-1, 1:-1]
