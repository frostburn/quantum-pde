# coding: utf-8
from __future__ import division

import argparse
import os
import sys
import tempfile

from matplotlib.animation import FuncAnimation
from pylab import *
import imageio

from lattice import make_lattice_2D, make_border_wall_2D, make_periodic_2D, RESOLUTIONS
from episodes import EPISODES
from tf_integrator import WaveFunction2D


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'Render video or animate the 2D Schrödinger equation')
    parser.add_argument('episode', help='Episode name. One of {}'.format(EPISODES.keys()))
    parser.add_argument('--resolution', help='Screen resolution. One of {}'.format(RESOLUTIONS.keys()), default='160p')
    parser.add_argument('--animate', help='Animate instead of dumping frames', action='store_true')
    parser.add_argument('--num_frames', help='Number of frames to dump', type=int, default=600)
    args = parser.parse_args()

    episode = EPISODES[args.episode](args.resolution)
    if isinstance(episode, tuple):
        dx, screen, psi, potential, episode_length = episode
    else:
        locals().update(episode)
    locals().setdefault("measurements", {})
    locals().setdefault("damping_field", None)

    t = 0
    dt = episode_length / args.num_frames

    extra_iterations = 0
    wave_function = WaveFunction2D(psi, potential, dx, dt, extra_iterations=extra_iterations, damping_field=damping_field)

    start = datetime.datetime.now()
    print("Rendering episode '{}'".format(args.episode))
    print("Resolution = {}".format(args.resolution))
    print("Episode length = {}".format(episode_length))
    print("Delta t = {}".format(dt))
    print("Ticks per frame = {}".format(wave_function.iterations))
    print("Start time = {}".format(start))

    def step():
        global t
        wave_function.step()
        t += dt
        for measurement_t in list(measurements.keys()):
            if t >= measurement_t:
                psi = wave_function.get_field()
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
                wave_function.set_field(psi)

    if args.animate:
        fig, ax = subplots()
        prob = abs(psi)**2
        impsi = imshow(prob[screen], vmin=0, vmax=0.001*prob.max())

        def init():
            return impsi,

        def update(frame):
            step()
            psi = wave_function.get_field()
            prob = abs(psi)**2
            if frame == 0:
                total_prob = prob.sum() * dx*dx
                print("t = {}, total probability = {}".format(t, total_prob))
            impsi.set_data(prob[screen])
            return impsi,

        ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True, repeat=True, interval=1)
        show()
    else:
        folder = tempfile.mkdtemp()
        writer = imageio.get_writer(os.path.join(folder, "out.mp4"), fps=60, quality=10)
        for frame in range(args.num_frames):
            step()
            rgb = np.minimum(wave_function.get_visual(screen).transpose(1, 2, 0) * 255, 255).astype('uint8')
            writer.append_data(rgb)

            if frame % 100 == 0:
                psi = wave_function.get_field()
                prob = abs(psi)**2
                total_prob = prob.sum() * dx*dx
                print("t = {}, total probability = {}".format(t, total_prob))
                if t > 0:
                    now = datetime.datetime.now()
                    duration = now - start
                    fraction_complete = t / episode_length
                    fraction_remaining = 1 - fraction_complete
                    remaining = datetime.timedelta(seconds=(duration.total_seconds() / fraction_complete * fraction_remaining))
                    eta = now + remaining
                    print("ETA = {}; {} left".format(eta, remaining))
                    print("")
        writer.close()
        print("Done. Results in {}".format(folder))
