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


def centered_fft2(psi):
    momentum = fft2(psi)
    w, h = momentum.shape
    w //= 2
    h //= 2
    momentum = concatenate((momentum[w:], momentum[:w]))
    momentum = concatenate((momentum[:, h:], momentum[:, :h]), axis=1)
    return momentum


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'Render video or animate the 2D Schrödinger equation')
    parser.add_argument('episode', help='Episode name. One of {}'.format(EPISODES.keys()))
    parser.add_argument('--resolution', help='Screen resolution. One of {}'.format(RESOLUTIONS.keys()), default='160p')
    parser.add_argument('--animate', help='Animate instead of dumping frames', action='store_true')
    parser.add_argument('--num-frames', help='Number of frames to dump', type=int, default=600)
    parser.add_argument('--output', help='Output filename for video')
    parser.add_argument('--contrast', help='Visual contrast', type=float, default=4.0)
    parser.add_argument('--potential-contrast', help='Visual contrast on the potential overlay', type=float, default=1.0)
    parser.add_argument('--hide-phase', help='Hide phase lines', action='store_true')
    parser.add_argument('--show-momentum', help='Show a fourier transform instead', action='store_true')
    parser.add_argument('--extra-iterations', help='Add extra precision to the integrator', type=int, default=0)
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

    wave_function = WaveFunction2D(psi, potential, dx, dt, extra_iterations=args.extra_iterations, damping_field=damping_field)

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
        if args.show_momentum:
            prob = abs(centered_fft2(psi))**2
        else:
            prob = abs(psi)**2
        impsi = imshow(prob[screen], vmin=0, vmax=args.contrast*0.1*prob.max())

        def init():
            return impsi,

        def update(frame):
            step()
            psi = wave_function.get_field()
            if args.show_momentum:
                prob = abs(centered_fft2(psi))**2
            else:
                prob = abs(psi)**2
            if frame == 0:
                total_prob = (abs(psi)**2).sum() * dx*dx
                print("t = {}, total probability = {}".format(t, total_prob))
            impsi.set_data(prob[screen])
            return impsi,

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
            rgb = wave_function.get_visual(
                screen,
                hide_phase=args.hide_phase,
                contrast=args.contrast,
                potential_contrast=args.potential_contrast,
                show_momentum=args.show_momentum
            )
            rgb = np.minimum(rgb.transpose(1, 2, 0) * 255, 255).astype('uint8')
            writer.append_data(rgb)

            if frame % 100 == 99:
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
        print("Done. Results in {}".format(output_filename))
