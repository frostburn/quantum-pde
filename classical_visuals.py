from __future__ import division

import argparse
import os

import scipy.misc
from pylab import *

from lattice import RESOLUTIONS
from classical_episodes import EPISODES

# ffmpeg -framerate 60 -i frame%05d.png -codec:v libx264 -crf 17 -preset slower -bf 2 -flags +cgop -pix_fmt yuv420p -movflags faststart out.mp4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'Render raw dumps of particle sample counts into video frames')
    parser.add_argument('folder', help='Folder with raw frames')
    parser.add_argument('--white_level', type=float, default=64.0, help='Number of particle samples per pixel to render as white')
    parser.add_argument('--episode', help='Episode to fetch potential from')
    parser.add_argument('--resolution', help='Screen resolution. One of {}'.format(RESOLUTIONS.keys()), default='160p')

    args = parser.parse_args()

    raw_folder = os.path.join(args.folder, 'raw')
    png_folder = os.path.join(args.folder, 'png')

    if not os.path.isdir(png_folder):
        os.mkdir(png_folder)

    if args.episode:
        potential = EPISODES[args.episode](1).get('potential')
        if potential is not None:
            width, height = RESOLUTIONS[args.resolution]
            scale = 0.1
            x = (arange(width) - width * 0.5) / height / scale
            y = (arange(height) - height * 0.5) / height / scale
            x, y = meshgrid(x, y)
            potential = potential(x, y)
    else:
        potential = None

    for filename in sorted(os.listdir(raw_folder)):
        print("Processing {}".format(filename))
        basename, _ = os.path.splitext(filename)
        with open(os.path.join(raw_folder, filename), "rb") as f:
            counts = load(f)
            if potential is None:
                img = scipy.misc.toimage(counts, cmin=0, cmax=args.white_level)
            else:
                counts = counts / args.white_level
                p = sqrt(potential)
                rgb = array([counts + 0.2 * p, counts + 0.7*p, counts + 0.4 * p])
                img = scipy.misc.toimage(rgb, cmin=0, cmax=1.0, channel_axis=0)
            img.save(os.path.join(png_folder, "{}.png".format(basename)))
