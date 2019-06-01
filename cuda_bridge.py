from pylab import *
from matplotlib.animation import FuncAnimation

import os
import imageio
import tempfile
import numpy as np
from subprocess import Popen, PIPE, check_output

from classical_episodes import EPISODES
from lattice import RESOLUTIONS

episode = "double_slit"
resolution = "480p"

width, height = RESOLUTIONS[resolution]
num_frames = 600
num_blocks = 100000
dt = 50.0 / num_frames
initial_distribution = 1
push = 0.3
force_type = 2


if episode == "double_slit":
    initial_distribution = 1
    push = 0.2
    force_type = 2

potential = EPISODES[episode](1)["potential"]
scale = 0.1
x = (arange(width) - width * 0.5) / height / scale
y = (arange(height) - height * 0.5) / height / scale
x, y = meshgrid(x, y)
potential = potential(x, y)

check_output([
    'nvcc', 'classical_particle.cu',
    '-DNUM_FRAMES={}'.format(num_frames),
    '-DNUM_BLOCKS={}'.format(num_blocks),
    '-DDT={}'.format(dt),
    '-DPUSH={}'.format(push),
    '-DWIDTH={}'.format(width),
    '-DHEIGHT={}'.format(height),
    '-DINITIAL_DISTRIBUTION={}'.format(initial_distribution),
    '-DFORCE_TYPE={}'.format(force_type),
])


with Popen("./a.out", stdout=PIPE) as p:
    writer = None
    if writer is None:
        folder = tempfile.mkdtemp()
        writer = imageio.get_writer(os.path.join(folder, "out.mp4"), fps=60, quality=10)
    for frame in range(num_frames):
        res = p.stdout.read(width*height*4)
        counts = np.frombuffer(res, dtype='int32').reshape(height, width) // 10
        rgb = np.minimum(255, np.array([counts + potential * 20, counts + 100 * potential, counts + potential * 30]).transpose(1,2,0)).astype('uint8')
        print(int(100*frame / num_frames))
        writer.append_data(rgb)
    print("Done. Results in {}",format(folder))
    if False:
        res = p.stdout.read(width*height*4)
        counts = np.frombuffer(res, dtype='int32').reshape(height, width)

        fig, ax = subplots()
        impsi = imshow(counts, vmin=0, vmax=1000, cmap='gray')

        def init():
            return impsi,

        def update(frame):
            res = p.stdout.read(width*height*4)
            counts = np.frombuffer(res, dtype='int32').reshape(height, width)
            impsi.set_data(counts)
            return impsi,

        ani = FuncAnimation(fig, update, frames=range(num_frames - 1), init_func=init, blit=True, repeat=False, interval=1)
        show()
