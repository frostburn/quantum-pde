from __future__ import division

import argparse
import os

from matplotlib.animation import FuncAnimation
import scipy.misc
from pylab import *
import numpy.random

from lattice import make_lattice_2D
from flow import schrodinger_flow_2D
from util import normalize_2D, advance_pde

from episodes import gaussian_superposition
from lattice import RESOLUTIONS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'Dump frames of wave function measurements to a folder')
    parser.add_argument('episode', help='Episode name')
    parser.add_argument('--folder', help='Folder to dump raw frames into')
    parser.add_argument('--resolution', help='Screen resolution. One of {}'.format(RESOLUTIONS.keys()), default='160p')
    parser.add_argument('--animate', help='Animate instead of dumping frames', action='store_true')
    parser.add_argument('--num_frames', help='Number of frames to dump', type=int, default=600)
    parser.add_argument('--radius', help='Radius of the measurement blib', type=float, default=0.08)
    args = parser.parse_args()

    if args.folder and args.animate:
        print('Animation is not supported while dumping to disk')
        sys.exit()

    if args.folder and not args.animate and not os.path.isdir(args.folder):
        print("Target folder doesn't exist")
        sys.exit()

    width, height = RESOLUTIONS[args.resolution]

    if args.episode == 'static_gaussian':
        x, y, dx, screen = make_lattice_2D(args.resolution, 5, 5)
        potential = 0*x
        psi = exp(-5*(x**2 + y**2)) + 0j
        psi = normalize_2D(psi[screen], dx)
        total_samples = 16000
    elif args.episode == 'gaussian_superposition':
        dx, screen, psi, potential, episode_length = gaussian_superposition(args.resolution)
        i = 0
        t = 0
        dt = 0.004 * dx
        while t < episode_length:
            if i % 100 == 0:
                print("Precalculating: {} % complete".format(int(100 * t / episode_length)))
            patch = advance_pde(t, psi, potential, dt, dx, schrodinger_flow_2D, dimensions=2)
            psi[4:-4, 4:-4] = patch
            t += dt
        psi = normalize_2D(psi[screen], dx)
        total_samples = 40000

    radius = args.radius * width * 0.1
    r = int(ceil(radius))
    rad_range = arange(2*r) - r

    counts = abs(psi) * 0

    p = abs(psi**2).flatten()
    p /= p.sum()

    def step():
        global counts
        for j in numpy.random.choice(len(p), total_samples // args.num_frames, p=p):
            k = j % psi.shape[1]
            j //= psi.shape[1]
            if radius < 1:
                counts[j, k] += 1
            else:
                off_a, off_b = rand(2)*2 - 1
                for a in rad_range:
                    for b in rad_range:
                        if (a + off_a)**2 + (b + off_b)**2 <= radius*radius:
                            if j+a >= 0 and j+a < counts.shape[0] and k+b >= 0 and k+b < counts.shape[1]:
                                counts[j+a, k+b] += maximum(0, 1 - (a*a + b*b) / (radius*radius))

    if args.animate:
        step()
        fig, ax = subplots()
        imc = imshow(counts, vmin=0, vmax=64, cmap='gray')

        def init():
            return imc,

        def update(frame):
            step()
            imc.set_data(counts)
            return imc,

        ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True, repeat=True, interval=1)
        show()
    else:
        png_path = os.path.join(args.folder, 'png')
        if not os.path.isdir(png_path):
            os.mkdir(png_path)
        for frame in range(args.num_frames):
            step()
            img = scipy.misc.toimage(counts, cmin=0, cmax=64)
            img.save(os.path.join(png_path, "frame{:05}.png".format(frame)))
