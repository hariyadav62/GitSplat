import json
import time
import requests
from instantsplat_gradio_api import process_scene
import numpy as np
import torch
from gaussian_renderer import render, GaussianModel
from scene import Scene
from arguments import ModelParams, PipelineParams
from scene.cameras import Camera
from utils.pose_utils import get_tensor_from_camera
import os
from argparse import ArgumentParser
from PIL import Image


import runpod
import os

network_volume_path = "/runpod-volume/"

BASE_URL = "http://213.173.105.84:50674/"


def process_data_async(input_dir, output_dir, n_views, iterations, user_id, device_id, project_id, photos_count):
    from datetime import datetime
    start_time = time.time()
    result = process_scene(input_dir, output_dir, n_views, iterations)
    end_time = time.time()
    processed_time = round((end_time - start_time) * 1000)
    print(f"Time taken: {processed_time}")

    if(result == "Success"):
        data = {
            'status': 'success',
            'message': 'Video generated successfully',
            'inputDir': input_dir,
            'outputDir': output_dir,
            'userId': user_id,
            'deviceId': device_id,
            'projectId': project_id,
            'photosCount': photos_count,
            'iterations': iterations,
            'processedTimeInMillis': processed_time
        }
    else:
        data = {
            'status': 'failure',
            'message': 'Video is not generated',
            'inputDir': input_dir,
            'outputDir': output_dir,
            'userId': user_id,
            'deviceId': device_id,
            'projectId': project_id,
            'photosCount': photos_count,
            'iterations': iterations,
            'processedTimeInMillis': processed_time
        }
    print("dats:", json.dumps(data))
    user_scene_process_result(data)
    return "Video generated successfully"

def user_scene_process_result(data):
    print(f"In user_scene_process_result() Data: {data}")
    url = BASE_URL + "InstantSplat/gaussianSplatController/userSceneProcessResult"
    headers = {'Content-Type': 'application/json'}
    print(f"URL: {url}")
    response = requests.post(url, data=json.dumps(data), headers=headers)

def user_image_generate_result(data):
    print(f"In user_scene_process_result() Data: {data}")
    url = BASE_URL + "InstantSplat/gaussianSplatController/userSceneProcessResult"
    headers = {'Content-Type': 'application/json'}
    print(f"URL: {url}")
    response = requests.post(url, data=json.dumps(data), headers=headers)

def generate_image(view_matrix_data, source_path, userId, deviceId, sentDeviceId, projectId):
    try:
        start_time = time.time()
        print(f"Received view_matrix_data: {view_matrix_data}")
        print(f"Received source_path: {source_path}")
        # Count the number of images in the specified directory
        images_path = os.path.join(source_path, "images")
        n_views = len([img for img in os.listdir(images_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Convert the view_matrix_data to a column-major numpy array
        user_view_matrix = np.array([
            [view_matrix_data['0'], view_matrix_data['4'], view_matrix_data['8'], view_matrix_data['12']],
            [view_matrix_data['1'], view_matrix_data['5'], view_matrix_data['9'], view_matrix_data['13']],
            [view_matrix_data['2'], view_matrix_data['6'], view_matrix_data['10'], view_matrix_data['14']],
            [view_matrix_data['3'], view_matrix_data['7'], view_matrix_data['11'], view_matrix_data['15']]
        ])
        # Transformation matrix for conversion
        T = np.array([
            [1, 0,  0, 0],
            [0, -1, 0, 0],  # Invert Y-axis
            [0, 0, -1, 0],  # Invert Z-axis
            [0, 0,  0, 1]
        ])

        # Convert the view matrix
        converted_view_matrix = T @ user_view_matrix

        T_x = np.array([
            [1, 0,  0, 0],
            [0, 1, 0, 0],  # Invert Y-axis
            [0, 0, 1, 0],  # Invert Z-axis
            [0, 0,  0, 1]
        ])
        converted_view_matrix_r = T_x @ converted_view_matrix
        # Extract rotation and translation from user provided matrix
        # R = np.eye(3)   
        R = converted_view_matrix_r[:3, :3]
        T = converted_view_matrix_r[:3, 3]
        print("Converted View matrix :", converted_view_matrix_r)

        # Initialize Model Parameters
        parser = ArgumentParser()
        params = ModelParams(parser)
        params.model_path = source_path+"/output/point_cloud/iteration_1000/"
        params.source_path = source_path
        params.cfg_args = os.path.join(params.model_path, "cfg_args")
        params.images = "images"
        params.resolution = -1
        params.white_background = False
        params.data_device = "cuda"
        params.eval = False
        params.n_views = n_views
        params.convert_SHs_python = False
        params.compute_cov3D_python = False
        params.debug = False
        params.iterations = 1000
        params.skip_train = False
        params.skip_test = False
        params.quiet = False
        params.optim_test_pose_iter = 500
        params.infer_video = True
        params.test_fps = False
        scene = Scene(params, GaussianModel(sh_degree=params.sh_degree))
        pipeline_params = PipelineParams(parser)        

        # Extract rotation and translation
        R = np.eye(3)
        T = user_view_matrix[:3, 3]

        base_camera = next(iter(scene.train_cameras.values()))[0]

        print("Base Camera FoVx:", base_camera.FoVx, base_camera.FoVy)
        virtual_camera = Camera(
            colmap_id=base_camera.colmap_id,
            R=R,
            T=T,
            FoVx=base_camera.FoVx,
            FoVy=base_camera.FoVy,
            image=base_camera.original_image,
            gt_alpha_mask=None,
            image_name=base_camera.image_name,
            uid=base_camera.uid
        )

        camera_pose = get_tensor_from_camera(virtual_camera.world_view_transform.transpose(0, 1))
        virtual_camera.world_view_transform = camera_pose

        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        final_image = render(virtual_camera, scene.gaussians, pipeline_params, bg_color, camera_pose=camera_pose)["render"]

        
        # Detach the tensor from the computation graph and convert to NumPy array
        final_image_np = final_image.detach().cpu().numpy()  # Detach and move to CPU before converting to NumPy array

        # Ensure the image is in the [0, 1] range, then scale to [0, 255] for image saving
        final_image_np = (final_image_np * 255).astype(np.uint8)

        # Convert the NumPy array to a PIL Image (Assuming the final_image_np shape is (H, W, C) or (C, H, W))
        # If the tensor is in the format (C, H, W), we need to transpose it to (H, W, C)
        if final_image_np.shape[0] == 3:  # (C, H, W) format
            final_image_np = final_image_np.transpose(1, 2, 0)  # Convert to (H, W, C)

        # Create the PIL Image
        final_image_pil = Image.fromarray(final_image_np)

        # Step 9: Rotate the Image by 180 degrees
        rotated_image_pil = final_image_pil.rotate(180)

        # Step 10: Save the Rotated Image using PIL's save method (not torchvision)
        # Ensure the generated_images directory exists
        generated_images_dir = os.path.join(source_path, 'generated_images')
        if not os.path.exists(generated_images_dir):
            os.makedirs(generated_images_dir)
        image_filename = os.path.join(generated_images_dir, f"rendered_from_ply.png")
        rotated_image_pil.save(image_filename)
        end_time = time.time()
        processed_time = round((end_time - start_time) * 1000)
        # Return the image file
        #return send_file(image_filename, mimetype='image/png')
        print(f"ImageFileName: {image_filename}")
        response = {
            'status': 'success',
            'message': 'Image Generated',
            'fileName': 'rendered_from_ply.png',
            'userId': userId,
            'deviceId': deviceId,
            'sentDeviceId': sentDeviceId,
            'projectId': projectId,
            'photosCount': n_views,
            'iterations': 1000, 
            'processedTimeInMillis': processed_time,   

        }
        # Return the image file name
        print(json.dumps(response))
        response = user_image_generate_result(response)
        return f"image generated successfully {userId}/{deviceId}/{projectId}"

    except Exception as e:
        return json.dumps({'status': 'error', 'message': str(e), 'fileName': 'NA'})




def handler(event):
    input = event['input']
    instruction = input.get('instruction')

    if instruction == "hello_world":
        return hello_world()
    
    if instruction == "generate_video":
        input_dir = input.get('input_dir')
        output_dir = input.get('output_dir')
        n_views = input.get('n_views')
        iterations = input.get('iterations')
        user_id = input.get('user_id')
        device_id = input.get('device_id')
        project_id = input.get('project_id')
        photos_count = input.get('photos_count')
        return process_data_async(input_dir, output_dir, n_views, iterations, user_id, device_id, project_id, photos_count)
    
    if instruction == "generate_image":
        view_matrix_data = input.get('view_matrix_data')
        source_path = input.get('source_path')
        userId = input.get('userId')
        deviceId = input.get('deviceId')
        sentDeviceId = input.get('sentDeviceId')
        projectId = input.get('projectId')
        return generate_image(view_matrix_data, source_path, userId, deviceId, sentDeviceId, projectId)

    return "Invalid instruction"

def hello_world():
    input_path = "assets/sora/Art"
    output_path = os.path.join(network_volume_path, "output")
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    output = process_scene(input_path, output_path, 3, 1000)
    return output

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})



    
    
# {
#   "input": {
#     "instruction": "generate_video",
#     "input_dir": "/runpod-volume/SplatFlies/PhotoFiles/1/1/1",
#     "output_dir": "/runpod-volume/SplatFlies/PhotoFiles/1/1/1/output1",
#     "n_views": 3,
#     "iterations": 1000,
#     "user_id": "1",
#     "device_id": "1",
#     "project_id": "1",
#     "photos_count": 3
#   }
# }
