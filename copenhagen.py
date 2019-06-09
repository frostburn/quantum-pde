from __future__ import division

import argparse
import os
import tempfile

from matplotlib.animation import FuncAnimation
import scipy.misc
from pylab import *
import numpy.random
import imageio

from lattice import make_lattice_2D
from flow import schrodinger_flow_2D
from util import normalize_2D, advance_pde

from episodes import gaussian_superposition
from lattice import RESOLUTIONS
from tf_integrator import WaveFunction2D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'Dump frames of wave function measurements to a folder')
    parser.add_argument('episode', help='Episode name')
    parser.add_argument('--resolution', help='Screen resolution. One of {}'.format(RESOLUTIONS.keys()), default='160p')
    parser.add_argument('--animate', help='Animate instead of dumping frames', action='store_true')
    parser.add_argument('--num-frames', help='Number of frames to dump', type=int, default=600)
    parser.add_argument('--radius', help='Radius of the measurement blib', type=float, default=0.08)
    parser.add_argument('--output', help='Output filename for video')
    args = parser.parse_args()

    width, height = RESOLUTIONS[args.resolution]

    if args.episode == 'static_gaussian':
        x, y, dx, screen = make_lattice_2D(args.resolution, 5, 5)
        potential = 0*x
        psi = exp(-3*(x**2 + y**2)) + 0j
        psi = normalize_2D(psi[screen], dx)
        total_samples = 16000
    elif args.episode == 'gaussian_superposition':
        episode = gaussian_superposition(args.resolution)
        locals().update(episode)
        i = 0
        t = 0
        dt = 0.004 * dx
        wave_function = WaveFunction2D(psi, potential, dx, dt, damping_field=damping_field)
        while t < episode_length:
            if i % 100 == 0:
                print("Precalculating: {} % complete".format(int(100 * t / episode_length)))
            wave_function.step()
            t += dt
        psi = normalize_2D(wave_function.get_field(), dx)
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
        output_filename = args.output
        if not output_filename:
            folder = tempfile.mkdtemp()
            output_filename = os.path.join(folder, "out.mp4")
        writer = imageio.get_writer(output_filename, fps=60, quality=10)
        for frame in range(args.num_frames):
            step()
            rgb = np.array([counts * 0.9, counts * 0.8, counts])
            if args.episode == "static_gaussian":
                rgb *= 0.02
            elif args.episode == "gaussian_superposition":
                rgb *= 0.1
            rgb = np.minimum(rgb.transpose(1, 2, 0) * 255, 255).astype('uint8')
            writer.append_data(rgb)
        writer.close()
        print("Done. Results in {}".format(output_filename))
