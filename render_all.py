import os
import subprocess
import tempfile
import shutil
from threading import Thread

from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np

from lattice import RESOLUTIONS

ASSETS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')

episodes = [
    # ["text", "Quantum mechanics explained\nusing an actual wave function", 30],
    ["text", u"Kvanttimekaniikka selitettynä\noikealla aaltofunktiolla", 30],  # 1
    ["tf_render", "harmonic_potential", 30, {'potential_contrast': 0.05}],  # 2
    ["tf_render", "static_gaussian", 30, {'hide_phase': True}],  # 3
    ["copenhagen", "static_gaussian", 30],  # 4
    ["tf_render", "static_gaussian", 30],  # 5
    ["classical_particle", "static_gaussian", 30],  # 6
    ["classical_particle", "superposition", 30],  # 7
    ["tf_render", "gaussian_superposition", 30], # 8
    ["copenhagen", "gaussian_superposition", 30],  # 9
    ["tf_render", "colliding_gaussians", 30, {'contrast': 6.0}],  # 10
    ["classical_particle", "colliding_superposition", 30],  # 11
    ["tf_render", "tunneling_slow", 30],  # 12
    ["tf_render", "tunneling_fast", 30],  # 13
    ["tf_render", "tunneling", 30],  # 14
    ["classical_particle", "tunneling", 30],  # 15
    ["tf_render", "single_slit", 30, {'contrast': 40.0}],  # 16
    ["tf_render", "double_slit", 30, {'contrast': 40.0}],  # 17
    ["classical_particle", "double_slit", 30],  # 18
    [[
        ["tf_render", "tunneling_slow"],
        ["tf_render", "tunneling_slow", {'show_momentum': True, 'contrast': 0.002}],
        ["classical_particle", "tunneling_slow"],
        ["classical_particle", "tunneling_slow", {"show_momentum": True}],
    ], 30],  # 19
    # ["fourier", 30],  # 20
    ["classical_particle", "square_measurement", 30],  # 21
    ["classical_particle", "square_measurement_inverted", 30],  # 22
    ["tf_render", "gaussian_measured_inverted", 30, {'contrast': 4.0}],  # 23
    ["tf_render", "gaussian_measured", 30, {'contrast': 4.0}],  # 24
    ["tf_render", "double_slit_measured", 30, {'contrast': 40.0}],  # 25
    ["tf_render", "convex_mirror", 30, {'contrast': 4.0}],  # 26
    ["classical_particle", "convex_mirror", 30],  # 27
    ["tf_render", "box_with_stuff", 30, {'contrast': 10.0}],  # 28
]

# timestamps = [
#     (-1, 0),
#     (0.1, 5.1),
#     (0.2, 42),
#     (1, 51.8),
#     (2, 60 + 14.5),
#     (3, 60 + 52),
#     (4, 140),
#     (5, 3*60+3),
#     (6, 3*60+23),
#     (7, 3*60+45),
#     (8, 4*60+8),
#     (9, 4*60+34),
#     (10, 4*60+44),
#     (11, 4*60+53),
#     (12, 5*60+19),
#     (13.1, 5*60+47),
#     (13.2, 6*60+12),
#     (14, 6*60+19),
#     (15, 6*60+40),
#     (16, 6*60+55),
#     (17, 8*60+2),
#     (18, 8*60+33),
#     (19, 9*60+8),
#     (20, 9*60+29),
#     (21, 9*60+49),
#     (22, 10*60+25),
# ]

# for i in range(len(timestamps) - 1):
#     episodes[i][2] = (timestamps[i+1][1] - timestamps[i][1])
#     assert episodes[i][2] > 0

base_resolution = "160p"
sub_resolution = str(int(base_resolution[:-1]) // 2) + "p"
width, height = RESOLUTIONS[base_resolution]
num_blocks = 1000
extra_iterations = 2
fps = 60
work_folder = "/home/lumi/quantum-pde-data"

frame = 0
remainder = 0.0

def render_episode(episode, num_frames, outfile, resolution=base_resolution):
    print(episode)
    print(outfile)
    hide_phase = False
    show_momentum = False
    potential_contrast = 1.0
    contrast = 4.0
    opts = locals()
    if len(episode) == 3:
        command, episode, num_seconds = episode
    elif len(episode) == 4:
        command, episode, num_seconds, options = episode
        potential_contrast = options.get("potential_contrast", potential_contrast)
        hide_phase = options.get("hide_phase", hide_phase)
        show_momentum = options.get("show_momentum", show_momentum)
        contrast = options.get("contrast", contrast)

    if command == "classical_particle":
        args = [
            "python", command + ".py", episode,
            "--output", outfile,
            "--resolution", resolution,
            "--num-frames", num_frames,
            "--num-blocks", num_blocks,
            "--contrast", contrast,
        ]
        if show_momentum:
            args.append("--show-momentum")
        subprocess.check_output(map(str, args))
    elif command == "tf_render":
        args = [
            "python", command + ".py", episode,
            "--output", outfile,
            "--resolution", resolution,
            "--num-frames", num_frames,
            "--potential-contrast", potential_contrast,
            "--extra-iterations", extra_iterations,
            "--contrast", contrast,
        ]
        if hide_phase:
            args.append("--hide-phase")
        if show_momentum:
            args.append("--show-momentum")
        subprocess.check_output(map(str, args))
    elif "copenhagen" in command:
        subprocess.check_output(map(str, [
            "python", command + ".py", episode,
            "--output", outfile,
            "--resolution", resolution,
            "--num-frames", num_frames,
        ]))
    else:
        raise ValueError("Unknown command {}".format(command))

for n, episode in enumerate(episodes):
    command = None
    if len(episode) == 2:
        sub_episodes, num_seconds = episode
    elif len(episode) == 3:
        command, arg, num_seconds = episode
    elif len(episode) == 4:
        command, arg, num_seconds, options = episode

    num_frames = fps * num_seconds + remainder
    remainder = num_frames - int(num_frames)
    num_frames = int(num_frames)

    outfile = os.path.join(work_folder, "episode{:02}.mp4".format(n))
    if os.path.isfile(outfile):
        print("File {} exists. Skipping...".format(outfile))
        continue

    if command is None:
        for m, sub_episode in enumerate(sub_episodes):
            sub_outfile = os.path.join(work_folder, "episode{:02}_{:02}.mp4".format(n, m))
            sub_args = sub_episode[:2] + [num_seconds] + sub_episode[2:]
            render_episode(sub_args, num_frames, sub_outfile, resolution=sub_resolution)
        subprocess.check_output(["touch", outfile])
        continue

    if command == "text":
        background = (100, 50, 30)
        fontsize = int(height * 0.1)
        font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf', fontsize)
        image = Image.new('RGB', (width, height), background)
        draw = ImageDraw.Draw(image)
        for i, text in enumerate(arg.split('\n')):
            w, h = font.getsize(text)
            draw.text(((width-w)//2, int((height-h)*0.5 + 1.5*(i-0.5)*h)), text, fill='white', font=font)
        writer = imageio.get_writer(outfile, fps=60, quality=10)
        for _ in range(num_frames):
            writer.append_data(np.array(image))
            frame += 1
        writer.close()
    else:
        render_episode(episode, num_frames, outfile)
