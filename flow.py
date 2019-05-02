# coding: utf-8
from util import laplacian2D


def schrodinger_flow2D(t, psi, potential, dx):
    """
    The flow part of the 2 dimensional Schrödinger equation.
    """
    return 1j * laplacian2D(psi, dx) - 1j * potential[1:-1, 1:-1] * psi[1:-1, 1:-1]
