import os
import subprocess
import tempfile
import shutil

episodes = [
    # ("schrodinger", "static_gaussian", 30),  # TODO: Needs manual shuffling and phase-line removal
    # ("copenhagen", "static_gaussian", 30),
    ("classical_particle", "superposition", 30),
    ("schrodinger2D", "gaussian_superposition", 30),
    # ("copenhagen", "gaussian_superposition", 30),
    ("schrodinger2D", "colliding_gaussians", 30),
    ("classical_particle", "colliding_superposition", 30),
    ("schrodinger2D", "tunneling_slow", 30),
    ("schrodinger2D", "tunneling_fast", 30),
    ("schrodinger2D", "tunneling", 30),
    ("classical_particle", "tunneling", 30),
    ("schrodinger2D", "single_slit", 30),
    ("schrodinger2D", "double_slit", 30),
    ("classical_particle", "double_slit", 30),
    ("classical_particle", "square_measurement", 30),
    ("classical_particle", "square_measurement_inverted", 30),
    ("schrodinger2D", "gaussian_measured", 30),
    ("schrodinger2D", "gaussian_measured_inverted", 30),
    ("schrodinger2D", "double_slit_measured", 30),
    ("schrodinger2D", "convex_mirror", 30),
    ("classical_particle", "convex_mirror", 30),
    ("schrodinger2D", "box_with_stuff", 30),
]

resolution = "80p"
sampling_multiplier = 1
white_level = 64 * sampling_multiplier
contrast = 16.0
fps = 60

master_folder = tempfile.mkdtemp()
frame = 0

for episode in episodes:
    command, episode, num_seconds = episode

    num_seconds = int(num_seconds * 0.1)

    folder = tempfile.mkdtemp()

    if "classical" in command:
        subprocess.call(map(str, [
            "python", command + ".py", episode,
            "--folder", folder,
            "--resolution", resolution,
            "--num_frames", (int(fps * num_seconds)),
            "--sampling_multiplier", sampling_multiplier,
        ]))
        subprocess.call(map(str, [
            "python", "classical_visuals.py", folder,
            "--white_level", white_level,
            "--episode", episode,
            "--resolution", resolution,
        ]))
    if "schrodinger2D" in command:
        subprocess.call(map(str, [
            "python", command + ".py", episode,
            "--folder", folder,
            "--resolution", resolution,
            "--num_frames", (int(fps * num_seconds)),
        ]))
        subprocess.call(map(str, [
            "python", "schrodinger2D_visuals.py", folder,
            "--contrast", contrast,
            "--episode", episode,
            "--resolution", resolution,
        ]))
    shutil.rmtree(os.path.join(folder, "raw"))
    for filename in sorted(os.listdir(os.path.join(folder, "png"))):
        source = os.path.join(folder, "png", filename)
        destination = os.path.join(master_folder, "frame{:05}.png".format(frame))
        shutil.move(source, destination)
        frame += 1
    shutil.rmtree(folder)

os.chdir(master_folder)
subprocess.call(
    "ffmpeg -framerate 60 -i frame%05d.png -codec:v libx264 -crf 17 -preset slower -bf 2 -flags +cgop -pix_fmt yuv420p -movflags faststart out.mp4",
    shell=True,
)
