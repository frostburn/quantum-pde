import scipy.misc
from pylab import *

# ffmpeg -i frame%05d.png -c:v libx264 -vf fps=60 -preset slower -crf 17 out.mp4

for frame in range(3200):
    print("Processing frame {}".format(frame))
    with open("./rendered/double_slit/frame{:05}.dat".format(frame), "rb") as f:
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
        img.save("./rendered/double_slit/png/frame{:05}.png".format(frame))
