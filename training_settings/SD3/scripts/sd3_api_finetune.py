import os
import json
import random
from urllib import request
import datetime
from PIL import Image, ImageDraw, ImageFont
import time
import re
import urllib.error

from dotenv import load_dotenv
load_dotenv()

# Configuration
api_workflow_dir = os.getenv("API_WORKFLOW_DIR")
finetune_dir = os.getenv("FINETUNE_DIR")

api_workflow_file = os.getenv("API_WORKFLOW_FILE")
api_endpoint = os.getenv("API_ENDPOINT")
image_output_dir = os.getenv("IMAGE_OUTPUT_DIR")
font_ttf_path = os.getenv("FONT_TTF_PATH")

comfyui_output_dir = os.getenv("COMFYUI_OUTPUT_DIR")

api_endpoint = f"http://{api_endpoint}/prompt"

workflow_file_path = os.path.join(api_workflow_dir, api_workflow_file)
workflow = json.load(open(workflow_file_path))

current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
relative_output_path = current_datetime

directory_creation_timeout = 3000  # Timeout for directory creation in seconds
image_generation_timeout = 30000  # Timeout for image generation in seconds

def get_checkpoint_number(filename):
    match = re.search(r'checkpoint-(\d+)', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'/checkpoint-(\d+)/', filename)
    if match:
        return int(match.group(1))
    return None

def get_most_recent_output_folder(base_dir):
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    if not folders:
        return None
    return max(folders, key=lambda f: os.path.getctime(os.path.join(base_dir, f)))

def process_safetensors(safetensor_dir, workflow):
    print(f"Scanning directory: {safetensor_dir}")
    
    last_dir = os.path.basename(os.path.normpath(safetensor_dir))
    
    all_items = os.listdir(safetensor_dir)
    
    safetensor_items = [f for f in all_items if f.endswith('.safetensors')]
    
    safetensor_items.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
    
    print(f"Found items: {safetensor_items}")
    
    for item in safetensor_items:
        unet_name = f"{last_dir}/{item}"
        
        print(f"Processing: {unet_name}")

        unet_loader_node = workflow["273"]
        unet_loader_node["inputs"]["unet_name"] = unet_name

        checkpoint_num = item.split('-')[1].split('.')[0]
        
        save_image = workflow["275"]
        filename_prefix = f"checkpoint-{checkpoint_num}"
        save_image["inputs"]["output_path"] = relative_output_path
        save_image["inputs"]["filename_prefix"] = filename_prefix

        success = queue_prompt(workflow)
        if not success:
            print(f"Failed to queue prompt for checkpoint {checkpoint_num}")
        else:
            print(f"Successfully queued prompt for checkpoint {checkpoint_num}")

    if not safetensor_items:
        print("No .safetensors files found in the directory.")
    
    return len(safetensor_items)

def create_image_strip(safetensor_dir, image_folder, output_filename):
    safetensor_files = [f for f in os.listdir(safetensor_dir) if f.endswith('.safetensors')]
    safetensor_files.sort(key=get_checkpoint_number)
    checkpoints = [get_checkpoint_number(f) for f in safetensor_files if get_checkpoint_number(f) is not None]

    images = []
    for checkpoint in checkpoints:
        filename = f"checkpoint-{checkpoint}_0001.png"
        filepath = os.path.join(image_folder, filename)
        if os.path.exists(filepath):
            try:
                img = Image.open(filepath)
                images.append(img)
            except IOError as e:
                print(f"Cannot open image: {filepath}")
                print(f"Error: {e}")

    if not images:
        print("No valid images found.")
        return

    img_width, img_height = images[0].size
    strip_width = img_width * len(images)
    label_height = 50  # Space for labels
    strip_height = img_height + label_height

    strip_image = Image.new('RGB', (strip_width, strip_height), 'white')
    draw = ImageDraw.Draw(strip_image)
    font = ImageFont.truetype(font_ttf_path, 20)

    for i, (img, checkpoint) in enumerate(zip(images, checkpoints)):
        strip_image.paste(img, (i * img_width, label_height))
        
        label = f"checkpoint-{checkpoint}"
        label_width = draw.textlength(label, font=font)
        label_x = i * img_width + (img_width - label_width) // 2
        draw.text((label_x, 10), label, fill="black", font=font)

    strip_image.save(output_filename)
    print(f"Image strip saved to: {output_filename}")

def queue_prompt(workflow):
    p = {"prompt": workflow}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(api_endpoint, data=data, headers={'Content-Type': 'application/json'})
    try:
        with request.urlopen(req) as response:
            print(f"API request successful. Status code: {response.getcode()}")
            return True
    except urllib.error.URLError as e:
        if hasattr(e, 'reason'):
            print(f"Failed to reach the server. Reason: {e.reason}")
        elif hasattr(e, 'code'):
            print(f"The server couldn't fulfill the request. Error code: {e.code}")
        print(f"API endpoint: {api_endpoint}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return False

def wait_for_directory_creation(directory, timeout):
    print(f"Waiting for directory {directory} to be created...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(directory):
            print(f"Directory {directory} found.")
            return True
        time.sleep(5)  # Check every 5 seconds
    print(f"Timeout waiting for directory {directory} to be created.")
    return False

def wait_for_images(image_folder, expected_count, timeout):
    print("Waiting for images to be generated...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(image_folder):
            image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
            if len(image_files) >= expected_count:
                print(f"Found all {expected_count} images.")
                return True
        time.sleep(5)  # Check every 5 seconds
    print("Timeout waiting for images to be generated.")
    return False

if __name__ == "__main__":
    safetensor_dir = finetune_dir 
    print(f"Finetune directory: {safetensor_dir}")

    # Generate images
    expected_image_count = process_safetensors(safetensor_dir, workflow)

    absolute_output_path = os.path.join(comfyui_output_dir, current_datetime)
    print(f"Absolute output path: {absolute_output_path}")

    # Create the image strip
    if wait_for_directory_creation(absolute_output_path, directory_creation_timeout):
        print(f"Expected image count: {expected_image_count}")
        if wait_for_images(absolute_output_path, expected_image_count, image_generation_timeout):
            output_strip_filename = os.path.join(absolute_output_path, "output_image_strip.png")
            create_image_strip(safetensor_dir, absolute_output_path, output_strip_filename)
        else:
            print("Failed to generate all images in time.")
    else:
        print("Output directory was not created.")