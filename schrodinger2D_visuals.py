import argparse
import os

import scipy.misc
from pylab import *

# ffmpeg -framerate 60 -i frame%05d.png -codec:v libx264 -crf 18 -preset slower -bf 2 -flags +cgop -pix_fmt yuv420p -movflags faststart out.mp4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'Render raw dumps of 2D wave function into video frames')
    parser.add_argument('folder', help='Folder with raw frames')

    args = parser.parse_args()

    raw_folder = os.path.join(args.folder, 'raw')
    png_folder = os.path.join(args.folder, 'png')

    if not os.path.isdir(png_folder):
        os.mkdir(png_folder)

    for filename in sorted(os.listdir(raw_folder)):
        print("Processing {}".format(filename))
        basename, _ = os.path.splitext(filename)
        with open(os.path.join(raw_folder, filename), "rb") as f:
            psi = load(f)
            phase = angle(psi)
            band1 = (phase / pi) ** 10
            band2 = ((1.05 + phase / pi) % 2 - 1) ** 16
            band3 = ((0.95 + phase / pi) % 2 - 1) ** 16
            rgb = array([band2, band3, band1])
            prob = abs(psi)**2
            prob /= 0.06
            prob = tanh(prob)
            rgb *= prob ** 0.5
            rgb += array([prob, prob, 0.8*prob])
            img = scipy.misc.toimage(rgb, cmin=0, cmax=1.0, channel_axis=0)
            img.save(os.path.join(png_folder, "{}.png".format(basename)))
