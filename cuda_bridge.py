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
resolution = "360p"

width, height = RESOLUTIONS[resolution]
num_frames = 400
num_blocks = 2000
exposure = 2
dt = 50.0 / num_frames
initial_distribution = 1
push = 0.3
force_type = 2


if episode == "double_slit":
    initial_distribution = 1
    push = 0.2
    force_type = 2

num_big_particles = 10
episode_params = EPISODES[episode](num_big_particles)
locals().update(episode_params)
scale = 0.1
x = (arange(width) - width * 0.5) / height / scale
y = (arange(height) - height * 0.5) / height / scale
x, y = meshgrid(x, y)
potential = potential(x, y)

check_output([
    'nvcc', 'classical_particle.cu',
    '-DNUM_FRAMES={}'.format(num_frames),
    '-DNUM_BLOCKS={}'.format(num_blocks),
    '-DDT={}'.format(dt / exposure),
    '-DPUSH={}'.format(push),
    '-DWIDTH={}'.format(width),
    '-DHEIGHT={}'.format(height),
    '-DINITIAL_DISTRIBUTION={}'.format(initial_distribution),
    '-DFORCE_TYPE={}'.format(force_type),
    '-DEXPOSURE={}'.format(exposure),
])


with Popen("./a.out", stdout=PIPE) as p:
    writer = None
    if writer is None:
        folder = tempfile.mkdtemp()
        writer = imageio.get_writer(os.path.join(folder, "out.mp4"), fps=60, quality=10)
    for frame in range(num_frames):
        res = p.stdout.read(width*height*4)
        counts = np.frombuffer(res, dtype='int32').reshape(height, width).astype('float64') / 2.0
        rgb = np.array([counts + potential * 20, counts + 100 * potential, counts * 0.8 + potential * 30])
        velocities += dt * force(positions)
        positions += dt * velocities
        for pos in positions:
            rgb += 100 * (1-tanh(-4 + 400*(x-pos[0])**2+400*(y-pos[1])**2))
        rgb = np.minimum(255, rgb.transpose(1,2,0)).astype('uint8')
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
