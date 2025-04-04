import subprocess
from pathlib import Path
import torch
import os
import sys

def run_process(cmd):
    print(f"Running command: {' '.join(cmd)}")
    env = os.environ.copy()

    # Ensure the virtual environment's packages are included
    env["PYTHONPATH"] = os.path.join(os.path.abspath("."), "instantsplat_env", "lib", "python3.11", "site-packages")

    # Debugging: Print the environment being used
    print(f"Running command: {' '.join(cmd)}")
    print(f"Using Python: {sys.executable}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, env=env)
    # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    return process.returncode == 0

def process_scene(input_dir, output_dir, n_views, iterations):
    torch.cuda.empty_cache()
    import sys
    print("Gradio is using Python:", sys.executable)

    if not torch.cuda.is_available():
        return "Error: CUDA not available"

    output_path = Path(output_dir)
    print(f"Output directory created at: {output_path}")

    init_cmd = [
        "python", "init_geo.py",
        "--source_path", input_dir,
        "--model_path", str(output_path),
        "--n_views", str(n_views),
        "--focal_avg",
        "--co_vis_dsp",
        "--conf_aware_ranking",
        "--infer_video"
    ]

    train_cmd = [
        "python", "train.py",
        "-s", input_dir,
        "-m", str(output_path),
        "--n_views", str(n_views),
        "--iterations", str(iterations),
        "--pp_optimizer",
        "--optim_pose"
    ]

    render_cmd = [
        "python", "render.py",
        "-s", input_dir,
        "-m", str(output_path),
        "--n_views", str(n_views),
        "--iterations", str(iterations),
        "--infer_video"
    ]

    commands = [
        (init_cmd, 0.2, "Initialization"),
        (train_cmd, 0.4, "Training"),
        (render_cmd, 0.8, "Rendering")
    ]

    for cmd, _, name in commands:
        if not run_process(cmd):
            return f"Error in {name}"
        print(f"After {name}, contents of output dir:")
        for item in os.listdir(output_path):
            print(f"  - {item}")

    video_path = output_path / "interp/ours_1000/interp_3_view.mp4"
    if video_path.exists():
        print(f"Found video at: {video_path}")
        return "Success"
    else:
        print("Video not found, searching for alternatives...")
        for mp4_file in output_path.rglob("*.mp4"):
            print(f"Found video at: {mp4_file}")
            #return str(mp4_file)
            return "Success"
    return "Video not found"
