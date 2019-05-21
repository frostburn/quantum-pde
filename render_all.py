import os
import subprocess
import tempfile
import shutil
from threading import Thread

from PIL import Image, ImageDraw, ImageFont

from lattice import RESOLUTIONS


episodes = [
    ["text", "Quantum mechanics explained\nusing an actual wave function", 30],
    ["schrodinger2D", "harmonic_potential", 30, {'contrast': 3.0}],
    ["schrodinger2D", "static_gaussian", 30, {'hide_phase': True, 'contrast': 4.0}],
    ["copenhagen", "static_gaussian", 30],
    ["schrodinger2D", "static_gaussian", 30],
    ["classical_particle", "superposition", 30],
    ["schrodinger2D", "gaussian_superposition", 30],
    ["copenhagen", "gaussian_superposition", 30],
    ["schrodinger2D", "colliding_gaussians", 30, {'contrast': 6.0}],
    ["classical_particle", "colliding_superposition", 30],
    ["schrodinger2D", "tunneling_slow", 30],
    ["schrodinger2D", "tunneling_fast", 30],
    ["schrodinger2D", "tunneling", 30],
    ["classical_particle", "tunneling", 30],
    ["schrodinger2D", "single_slit", 30, {'contrast': 40.0}],
    ["schrodinger2D", "double_slit", 30, {'contrast': 40.0}],
    ["classical_particle", "double_slit", 30, {'sampling_multiplier': 10.0}],
    ["classical_particle", "square_measurement", 30],
    ["classical_particle", "square_measurement_inverted", 30],
    ["schrodinger2D", "gaussian_measured_inverted", 30, {'contrast': 4.0}],
    ["schrodinger2D", "gaussian_measured", 30, {'contrast': 4.0}],
    ["schrodinger2D", "double_slit_measured", 30, {'contrast': 40.0}],
    ["schrodinger2D", "convex_mirror", 30, {'contrast': 4.0}],
    ["classical_particle", "convex_mirror", 30],
    ["schrodinger2D", "box_with_stuff", 30, {'contrast': 10.0}],
]

timestamps = [
    (-1, 0),
    (0.1, 5.1),
    (0.2, 42),
    (1, 51.8),
    (2, 60 + 14.5),
    (3, 60 + 52),
    (4, 140),
    (5, 3*60+3),
    (6, 3*60+23),
    (7, 3*60+45),
    (8, 4*60+8),
    (9, 4*60+34),
    (10, 4*60+44),
    (11, 4*60+53),
    (12, 5*60+19),
    (13.1, 5*60+47),
    (13.2, 6*60+12),
    (14, 6*60+19),
    (15, 6*60+40),
    (16, 6*60+55),
    (17, 8*60+2),
    (18, 8*60+33),
    (19, 9*60+8),
    (20, 9*60+29),
    (21, 9*60+49),
    (22, 10*60+25),
]

for i in range(len(timestamps) - 1):
    episodes[i][2] = (timestamps[i+1][1] - timestamps[i][1])
    assert episodes[i][2] > 0

resolution = "360p"
width, height = RESOLUTIONS[resolution]
default_sampling_multiplier = 3.0
white_level = 128
default_contrast = 8.0
fps = 60

master_folder = tempfile.mkdtemp()
frame = 0
remainder = 0.0

def render_episode(episode, folder, num_frames):
    print(episode)
    if len(episode) == 3:
        command, episode, num_seconds = episode
        hide_phase = False
        contrast = default_contrast
        sampling_multiplier = default_sampling_multiplier
    elif len(episode) == 4:
        command, episode, num_seconds, options = episode
        hide_phase = options.get('hide_phase', False)
        contrast = options.get('contrast', default_contrast)
        sampling_multiplier = options.get('sampling_multiplier', default_sampling_multiplier)

    if "classical" in command:
        subprocess.call(map(str, [
            "python", command + ".py", episode,
            "--folder", folder,
            "--resolution", resolution,
            "--num_frames", num_frames,
            "--sampling_multiplier", sampling_multiplier,
        ]))
        subprocess.call(map(str, [
            "python", "classical_visuals.py", folder,
            "--white_level", white_level,
            "--episode", episode,
            "--resolution", resolution,
        ]))
        shutil.rmtree(os.path.join(folder, "raw"))
    elif "schrodinger2D" in command:
        subprocess.call(map(str, [
            "python", command + ".py", episode,
            "--folder", folder,
            "--resolution", resolution,
            "--num_frames", num_frames,
        ]))
        cmd = [
            "python", "schrodinger2D_visuals.py", folder,
            "--contrast", contrast,
            "--episode", episode,
            "--resolution", resolution,
        ]
        if hide_phase:
            cmd.append('--hide_phase')
        subprocess.call(map(str, cmd))
        shutil.rmtree(os.path.join(folder, "raw"))
    elif "copenhagen" in command:
        subprocess.call(map(str, [
            "python", command + ".py", episode,
            "--folder", folder,
            "--resolution", resolution,
            "--num_frames", num_frames,
        ]))
    else:
        raise ValueError("Unknown command {}".format(command))

threads = []
folders = []
for episode in episodes:
    if len(episode) == 3:
        command, arg, num_seconds = episode
        hide_phase = False
    elif len(episode) == 4:
        command, arg, num_seconds, hide_phase = episode

    folder = tempfile.mkdtemp()

    num_frames = fps * num_seconds + remainder
    remainder = num_frames - int(num_frames)
    num_frames = int(num_frames)

    if command == "text":
        background = (100, 50, 30)
        fontsize = int(height * 0.1)
        font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf', fontsize)
        image = Image.new('RGB', (width, height), background)
        draw = ImageDraw.Draw(image)
        for i, text in enumerate(arg.split('\n')):
            w, h = font.getsize(text)
            draw.text(((width-w)//2, int((height-h)*0.5 + 1.5*(i-0.5)*h)), text, fill='white', font=font)
        for _ in range(num_frames):
            image.save(os.path.join(master_folder, "frame{:05}.png".format(frame)))
            frame += 1
    else:
        folders.append(folder)
        threads.append(Thread(target=render_episode, args=(episode, folder, num_frames)))
        threads[-1].start()

for folder, thread in zip(folders, threads):
    thread.join()
    for filename in sorted(os.listdir(os.path.join(folder, "png"))):
        source = os.path.join(folder, "png", filename)
        destination = os.path.join(master_folder, "frame{:05}.png".format(frame))
        shutil.move(source, destination)
        frame += 1
    shutil.rmtree(folder)

os.chdir(master_folder)
subprocess.call(
    "ffmpeg -framerate 60 -i frame%05d.png -i /home/lumi/Music/Quantum\\ Mechanics/part1_edited.wav -codec:v libx264 -codec:a aac -crf 17 -preset slower -bf 2 -flags +cgop -pix_fmt yuv420p -movflags faststart out.mp4",
    shell=True,
)
