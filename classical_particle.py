from pylab import *
from matplotlib.animation import FuncAnimation

import argparse
import os
import imageio
import tempfile
import numpy as np
from subprocess import Popen, PIPE, check_output

from classical_episodes import EPISODES
from lattice import RESOLUTIONS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'Render video or animate a classical particle distribution')
    parser.add_argument('episode', help='Episode name. One of {}'.format(EPISODES.keys()))
    parser.add_argument('--resolution', help='Screen resolution. One of {}'.format(RESOLUTIONS.keys()), default='160p')
    parser.add_argument('--animate', help='Animate instead of dumping frames', action='store_true')
    parser.add_argument('--num-frames', help='Number of frames to dump', type=int, default=600)
    parser.add_argument('--num-blocks', help='Number of 1k particle blocks to add to the cloud', type=int, default=1000)
    parser.add_argument('--output', help='Output filename for video')
    parser.add_argument('--contrast', help='Visual contrast', type=float, default=1.0)
    parser.add_argument('--exposure', help='Extra inter-frame iterations', type=int, default=2)
    parser.add_argument('--num-big-particles', help='Number of overlaid big particles', type=int, default=20)
    parser.add_argument('--show-momentum', help='Render speed instead of position', action='store_true')
    parser.add_argument('--seed', help='Seed for numpy RNG', type=int)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    episode = args.episode

    shift = 0
    push = 0
    force_type = 0
    measurement_type = 0
    if episode == "double_slit":
        initial_distribution = 1
        shift = -5
        push = 0.2
        force_type = 2
    elif episode.startswith("tunneling"):
        initial_distribution = 1
        shift = -4
        push = 0.75
        force_type = 1
        if episode == "tunneling_slow":
            push = 0.3
    elif episode == "static_gaussian":
        initial_distribution = 1
    elif episode == "superposition":
        initial_distribution = 2
    elif episode == "colliding_superposition":
        initial_distribution = 2
        push = 0.5
    elif episode == "square_measurement":
        initial_distribution = 1
        measurement_type = 1
    elif episode == "square_measurement_inverted":
        initial_distribution = 1
        measurement_type = 2
    elif episode == "convex_mirror":
        initial_distribution = 1
        shift = -4
        push = 0.75
        force_type = 3
    else:
        raise ValueError("Unknown episode: {}".format(episode))

    episode_params = EPISODES[episode](args.num_big_particles)
    locals().update(episode_params)
    locals().setdefault("potential", lambda x, y: 0*x)
    locals().setdefault("force", lambda pos: 0*pos)
    locals().setdefault("measurements", {})
    dt = episode_length / args.num_frames
    scale = 0.1
    width, height = RESOLUTIONS[args.resolution]
    x = (arange(width) - width * 0.5) / height / scale
    y = (arange(height) - height * 0.5) / height / scale
    x, y = meshgrid(x, y)
    potential = potential(x, y)
    t = 0

    if args.show_momentum:
        potential *= 0

    check_output([
        'nvcc', 'classical_particle.cu',
        '-DNUM_FRAMES={}'.format(args.num_frames),
        '-DNUM_BLOCKS={}'.format(args.num_blocks),
        '-DDT={}'.format(dt / args.exposure),
        '-DPUSH={}'.format(push),
        '-DSHIFT={}'.format(shift),
        '-DWIDTH={}'.format(width),
        '-DHEIGHT={}'.format(height),
        '-DINITIAL_DISTRIBUTION={}'.format(initial_distribution),
        '-DFORCE_TYPE={}'.format(force_type),
        '-DEXPOSURE={}'.format(args.exposure),
        '-DSHOW_MOMENTUM={}'.format(int(args.show_momentum)),
        '-DMEASUREMENT_TYPE={}'.format(measurement_type),
    ])


    with Popen("./a.out", stdout=PIPE) as p:
        output_filename = args.output
        if not output_filename:
            folder = tempfile.mkdtemp()
            output_filename = os.path.join(folder, "out.mp4")
        writer = imageio.get_writer(output_filename, fps=60, quality=10)
        for frame in range(args.num_frames):
            t += dt
            for measurement_t in list(measurements.keys()):
                if t >= measurement_t:
                    condition = measurements.pop(measurement_t)
                    keepers = condition(positions)
                    positions[~keepers] += 10
                    velocities[~keepers] = 0
            res = p.stdout.read(width*height*4)
            counts = np.frombuffer(res, dtype='int32').reshape(height, width).astype('float64') * (0.01 * args.contrast * width * height / args.num_blocks)
            rgb = np.array([potential * 20, potential * 100, potential * 30])
            rgb += np.array([counts, counts, counts * 0.8])
            rgb = np.minimum(rgb, 240)
            velocities += dt * force(positions)
            positions += dt * velocities
            if args.show_momentum:
                for vel in velocities:
                    rgb += 100 * (1-tanh(-4 + 400*(x-10*vel[0])**2+400*(y-10*vel[1])**2))
            else:
                for pos in positions:
                    rgb += 100 * (1-tanh(-4 + 400*(x-pos[0])**2+400*(y-pos[1])**2))
            rgb = np.minimum(255, rgb.transpose(1,2,0)).astype('uint8')
            writer.append_data(rgb)
            if frame % 100 == 99:
                print("{} % complete".format(int(100*frame / args.num_frames)))
        writer.close()
        print("Done. Results in {}".format(output_filename))
