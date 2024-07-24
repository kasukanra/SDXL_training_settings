#!/bin/bash

# Define source and target directories
SOURCE_DIR="/home/pure_water_100/kohya_train/sd3/sd3_medium_full_11"

TARGET_DIR="/mnt/c/home/ComfyUI/models/checkpoints/sd3_adam_11"

# Create the target directory if it doesn't already exist
mkdir -p "$TARGET_DIR"

# Find all .safetensors files in the source directory and symlink them to the target directory
find "$SOURCE_DIR" -type f -name "*.safetensors" -exec ln -s {} "$TARGET_DIR" \;

echo "Symlinks created for all .safetensors files from $SOURCE_DIR to $TARGET_DIR."