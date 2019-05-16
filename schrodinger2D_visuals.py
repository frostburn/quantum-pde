import argparse
import os

import scipy.misc
from pylab import *

from lattice import RESOLUTIONS
from episodes import EPISODES

# ffmpeg -framerate 60 -i frame%05d.png -codec:v libx264 -crf 17 -preset slower -bf 2 -flags +cgop -pix_fmt yuv420p -movflags faststart out.mp4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'Render raw dumps of 2D wave function into video frames')
    parser.add_argument('folder', help='Folder with raw frames')
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--contrast', type=float, default=16.0)
    parser.add_argument('--hide_phase', action='store_true')
    parser.add_argument('--episode', help='Episode to fetch potential from')
    parser.add_argument('--resolution', help='Screen resolution. One of {}'.format(RESOLUTIONS.keys()), default='160p')

    args = parser.parse_args()

    raw_folder = os.path.join(args.folder, 'raw')
    png_folder = os.path.join(args.folder, 'png')

    if not os.path.isdir(png_folder):
        os.mkdir(png_folder)

    if args.episode:
        episode = EPISODES[args.episode](args.resolution)
        if isinstance(episode, tuple):
            dx, screen, psi, potential, episode_length = episode
        else:
            potential = episode["potential"]
            screen = episode["screen"]
        potential = potential[screen]
        potential += potential.min()
        norm = potential.max()
        if norm:
            potential /= potential.max()
    else:
        potential = None

    for filename in sorted(os.listdir(raw_folder)):
        print("Processing {}".format(filename))
        basename, _ = os.path.splitext(filename)
        with open(os.path.join(raw_folder, filename), "rb") as f:
            psi = load(f)
            if args.mirror:
                psi = 0.5 * (psi + psi[:,::-1])
            phase = angle(psi)
            band1 = (phase / pi) ** 10
            band2 = ((1.05 + phase / pi) % 2 - 1) ** 16
            band3 = ((0.95 + phase / pi) % 2 - 1) ** 16
            rgb = array([band2, band3, band1])
            if args.hide_phase:
                rgb *= 0
            prob = abs(psi)**2
            prob *= args.contrast
            prob = tanh(prob)
            rgb *= prob ** 0.8
            rgb += array([prob, prob, 0.7*prob])
            if potential is not None:
                rgb += array([potential*0.2, potential*0.7, potential*0.4])
            img = scipy.misc.toimage(rgb, cmin=0, cmax=1.0, channel_axis=0)
            img.save(os.path.join(png_folder, "{}.png".format(basename)))
