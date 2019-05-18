from __future__ import division

import argparse
import datetime
import os

from matplotlib.animation import FuncAnimation

from pylab import *
import numpy.random

from lattice import RESOLUTIONS
from classical_episodes import EPISODES

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'Dump frames of classical particle distribution to a folder')
    parser.add_argument('episode', help='Episode name. One of {}'.format(EPISODES.keys()))
    parser.add_argument('--folder', help='Folder to dump raw frames into')
    parser.add_argument('--resolution', help='Screen resolution. One of {}'.format(RESOLUTIONS.keys()), default='160p')
    parser.add_argument('--animate', help='Animate instead of dumping frames', action='store_true')
    parser.add_argument('--num_frames', help='Number of frames to dump', type=int, default=600)
    parser.add_argument('--sampling_multiplier', help='Increase the number of samples taken', type=float, default=1)
    args = parser.parse_args()

    if args.folder and args.animate:
        print('Animation is not supported while dumping to disk')
        sys.exit()

    if args.folder and not args.animate and not os.path.isdir(args.folder):
        print("Target folder doesn't exist")
        sys.exit()

    width, height = RESOLUTIONS[args.resolution]

    num_particles = int(width * height * args.sampling_multiplier)

    scale = 0.1

    episode = EPISODES[args.episode](num_particles)
    locals().update(episode)

    locals().setdefault("force", lambda pos: pos*0)
    locals().setdefault("measurements", {})

    t = 0.0
    dt = episode_length / args.num_frames

    counts = zeros((height, width), dtype=int)

    start = datetime.datetime.now()

    print("Rendering episode '{}'".format(args.episode))
    print("Resolution = {}".format(args.resolution))
    print("Num frames = {}".format(args.num_frames))
    print("Delta t = {}".format(dt))
    print("Start time = {}".format(start))

    def step():
        global t, positions, velocities, phases, counts
        for measurement_t in list(measurements.keys()):
            if t >= measurement_t:
                condition = measurements.pop(measurement_t)
                keepers = condition(positions)
                positions[~keepers] += 10
                velocities[~keepers] = 0
        counts *= 0
        for pos in positions:
            i = int(floor(pos[0] * height * scale + width * 0.5))
            j = int(floor(pos[1] * height * scale + height * 0.5))
            if i >= 0 and i < width and j >= 0 and j < height:
                counts[j, i] += 1
        velocities += dt * force(positions)
        positions += dt * velocities
        t += dt

    if args.animate:
        step()
        fig, ax = subplots()
        imc = imshow(abs(counts), vmin=0, vmax=50 * args.sampling_multiplier, cmap='gray')

        def init():
            return imc,

        def update(frame):
            if frame == 0:
                print("t = {}".format(t))
            step()
            imc.set_data(abs(counts))
            return imc,

        ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True, repeat=True, interval=1)
        show()
    else:
        raw_path = os.path.join(args.folder, 'raw')
        if not os.path.isdir(raw_path):
            os.mkdir(raw_path)
        for frame in range(args.num_frames):
            step()
            with open(os.path.join(raw_path, "frame{:05}.dat".format(frame)), "wb") as f:
                save(f, counts)
            if frame % 100 == 99:
                now = datetime.datetime.now()
                duration = now - start
                fraction_complete = t / episode_length
                fraction_remaining = 1 - fraction_complete
                remaining = datetime.timedelta(seconds=(duration.total_seconds() / fraction_complete * fraction_remaining))
                eta = now + remaining
                print("ETA = {}; {} left".format(eta, remaining))
                print("")
