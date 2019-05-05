from __future__ import division
import numpy as np

RESOLUTIONS = {
    "2160p": (3840, 2160),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
    "360p": (640, 360),
    "240p": (426, 240),
    "160p": (284, 160),
    "80p": (142, 80),
}


def make_lattice_2D(resolution, box_width, extra, stencil=4):
    width, height = RESOLUTIONS[resolution]
    aspect_ratio = width / height
    dx = box_width / width
    total_box_width = box_width + 2 * extra
    total_box_height = box_width / aspect_ratio + 2 * extra
    total_width = int(np.ceil(total_box_width / dx)) + stencil * 2
    total_height = int(np.ceil(total_box_height / dx)) + stencil * 2
    x = (np.arange(total_width) - stencil) * dx - extra - box_width * 0.5
    y = (np.arange(total_height) - stencil) * dx - extra - box_width * 0.5 / aspect_ratio

    x, y = np.meshgrid(x, y)

    offscreen = stencil + int(np.ceil(extra / dx))
    screen_slice = (slice(offscreen, offscreen + height), slice(offscreen, offscreen + width))

    return x, y, dx, screen_slice


def make_border_wall_2D(resolution, box_width, extra, weight=1000, stencil=4):
    width, height = RESOLUTIONS[resolution]
    aspect_ratio = width / height
    x, y, dx, _ = make_lattice_2D(resolution, box_width, extra, stencil=stencil)
    r = box_width * 0.5
    x = np.where(x < -r, ((x+r)/extra)**2, np.where(x > r, ((x-r)/extra)**2, 0))
    r /= aspect_ratio
    y = np.where(y < -r, ((y+r)/extra)**2, np.where(y > r, ((y-r)/extra)**2, 0))
    return np.maximum(x, y) * weight


def make_periodic_2D(psi, stencil=4):
    """
    Copies parts of a 2D array so that it appears periodic up to `stencil` width.
    """
    near = slice(None, stencil)
    far = slice(-stencil, None)
    here = slice(stencil, 2*stencil)
    there = slice(-2*stencil, -stencil)
    # Edges
    psi[near, :] = psi[there, :]
    psi[far, :] = psi[here, :]
    psi[:, near] = psi[:, there]
    psi[:, far] = psi[:, here]
    # Corners
    psi[near, near] = psi[there, there]
    psi[near, far] = psi[there, here]
    psi[far, near] = psi[here, there]
    psi[far, far] = psi[here, here]
