# coding: utf-8
from __future__ import division

import argparse
import os
import sys


from matplotlib.animation import FuncAnimation
from pylab import *

from flow import schrodinger_flow_2D
from util import normalize_2D, advance_pde
from lattice import make_lattice_2D, make_border_wall_2D, make_periodic_2D, RESOLUTIONS
from episodes import EPISODES


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'Dump frames of 2D Schrödinger equation to a folder')
    parser.add_argument('episode', help='Episode name. One of {}'.format(EPISODES.keys()))
    parser.add_argument('--folder', help='Folder to dump raw frames into')
    parser.add_argument('--resolution', help='Screen resolution. One of {}'.format(RESOLUTIONS.keys()), default='160p')
    parser.add_argument('--animate', help='Animate instead of dumping frames', action='store_true')
    parser.add_argument('--num_frames', help='Number of frames to dump', type=int, default=600)
    args = parser.parse_args()

    if args.folder and args.animate:
        print('Animation is not supported while dumping to disk')
        sys.exit()

    if args.folder and not args.animate and not os.path.isdir(args.folder):
        print("Target folder doesn't exist")
        sys.exit()

    episode = EPISODES[args.episode](args.resolution)
    if isinstance(episode, tuple):
        dx, screen, psi, potential, episode_length = episode
    else:
        locals().update(episode)
    locals().setdefault("measurements", {})

    t = 0

    dt_frame = episode_length / args.num_frames
    ticks_per_frame = 1

    while dt_frame > 0.004 * dx * ticks_per_frame:
        ticks_per_frame += 1

    dt = dt_frame / ticks_per_frame

    start = datetime.datetime.now()

    print("Rendering episode '{}'".format(args.episode))
    print("Resolution = {}".format(args.resolution))
    print("Episode length = {}".format(episode_length))
    print("Delta t = {}".format(dt))
    print("Ticks per frame = {}".format(ticks_per_frame))
    print("Start time = {}".format(start))

    def step():
        global psi, t
        for j in range(ticks_per_frame):
            patch = advance_pde(t, psi, potential, dt, dx, schrodinger_flow_2D, dimensions=2)
            psi[4:-4, 4:-4] = patch
            t += dt
            for measurement_t in list(measurements.keys()):
                if t >= measurement_t:
                    measurement = measurements.pop(measurement_t)
                    mask = measurement["mask"]
                    if measurement["forced"]:
                        psi *= mask
                    else:
                        psi = normalize_2D(psi, dx)
                        prob = (abs(psi * mask)**2*dx*dx).sum()
                        if prob > rand():
                            psi *= mask
                        else:
                            psi *= 1-mask
        psi = normalize_2D(psi, dx)

    if args.animate:
        fig, ax = subplots()
        prob = abs(psi)**2
        impsi = imshow(prob[screen], vmin=0, vmax=0.1*prob.max())

        def init():
            return impsi,

        def update(frame):
            global psi, t
            if frame == 0:
                print("t = {}, Energy = {}".format(t, (dx*dx*abs(schrodinger_flow_2D(0, psi, potential, dx))**2).sum()))
            step()
            prob = abs(psi)**2
            impsi.set_data(prob[screen])
            return impsi,

        ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True, repeat=True, interval=1)
        show()
    else:
        checkpoint_path = os.path.join(args.folder, 'checkpoint')
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        raw_path = os.path.join(args.folder, 'raw')
        if not os.path.isdir(raw_path):
            os.mkdir(raw_path)
        for frame in range(args.num_frames):
            if frame % 100 == 0:
                print("t = {}, Energy = {}".format(t, (dx*dx*abs(schrodinger_flow_2D(0, psi, potential, dx))**2).sum()))
                if t > 0:
                    now = datetime.datetime.now()
                    duration = now - start
                    fraction_complete = t / episode_length
                    fraction_remaining = 1 - fraction_complete
                    remaining = datetime.timedelta(seconds=(duration.total_seconds() / fraction_complete * fraction_remaining))
                    eta = now + remaining
                    print("ETA = {}; {} left".format(eta, remaining))
                    print("")
            if frame % 1000 == 0:
                with open(os.path.join(checkpoint_path, "frame{:05}.dat".format(frame)), "wb") as f:
                    save(f, psi)
            with open(os.path.join(raw_path, "frame{:05}.dat".format(frame)), "wb") as f:
                save(f, psi[screen])
            step()
            frame += 1


