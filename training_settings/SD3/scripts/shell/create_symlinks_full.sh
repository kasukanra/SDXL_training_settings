#!/bin/bash

# Source directory where the models are stored
SOURCE_DIR="/home/pure_water_100/simpletuner_models/sd3_09/datasets"

# Target directory for symlinks
TARGET_DIR="/mnt/c/home/ComfyUI/models/unet/simpletuner_09"

# Iterate over each checkpoint directory
for CHECKPOINT_DIR in $(ls -d ${SOURCE_DIR}/checkpoint-*); do
    # Extract the checkpoint number from the directory name
    CHECKPOINT_NAME=$(basename ${CHECKPOINT_DIR})
    
    # Define the source file path
    SOURCE_FILE="${CHECKPOINT_DIR}/transformer/diffusion_pytorch_model.safetensors"
    
    # Define the symlink name
    LINK_NAME="${TARGET_DIR}/${CHECKPOINT_NAME}.safetensors"
    
    # Check if the source file exists
    if [ -f "${SOURCE_FILE}" ]; then
        # Create a symlink in the target directory
        ln -s "${SOURCE_FILE}" "${LINK_NAME}"
        echo "Symlink created for ${CHECKPOINT_NAME}"
    else
        echo "File not found: ${SOURCE_FILE}"
    fi
done

echo "Symlinking complete."