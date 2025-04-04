import gradio as gr
import subprocess
from pathlib import Path
import torch
import os
import sys

def run_process(cmd):
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    return process.returncode == 0

def process_scene(input_dir, output_dir, n_views, iterations, progress=gr.Progress()):
    import sys
    print("Gradio is using Python:", sys.executable)

    if not torch.cuda.is_available():
        return "Error: CUDA not available"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
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

    for cmd, prog, name in commands:
        progress(prog, f"Running {name}...")
        if not run_process(cmd):
            print("run process not success")
            return f"Error in {name}"
        print(f"After {name}, contents of output dir:")
        for item in os.listdir(output_path):
            print(f"  - {item}")

    video_path = output_path / "interp/ours_1000/interp_3_view.mp4"
    if video_path.exists():
        print(f"Found video at: {video_path}")
        return str(video_path)
    else:
        print("Video not found, searching for alternatives...")
        for mp4_file in output_path.rglob("*.mp4"):
            print(f"Found video at: {mp4_file}")
            return str(mp4_file)
    return "Video not found"

with gr.Blocks() as demo:
    gr.Markdown("# InstantSplat Demo")
    with gr.Row():
        with gr.Column():
            input_dir = gr.Textbox(label="Input Directory")
            output_dir = gr.Textbox(label="Output Directory")
            n_views = gr.Dropdown(choices=[2, 3, 4, 5, 6, 7, 8, 9], value=3, label="Number of Views")
            iterations = gr.Slider(minimum=1000, maximum=30000, value=1000, step=1000, label="Training Iterations")
            process_btn = gr.Button("Process Scene")
        with gr.Column():
            output_video = gr.Video(label="Output Video")

    process_btn.click(fn=process_scene, inputs=[input_dir, output_dir, n_views, iterations], outputs=output_video)

if __name__ == "__main__":
    # demo.launch()
    # Add the API route for process_scene
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
