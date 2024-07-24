#!/bin/bash

# Set the directory path
DIR="/mnt/e/datasets/SDXL/full_dataset_neo"

# Find and remove all *_sd3.npz files
find "$DIR" -type f -name "*_sd3.npz" -print -delete

echo "All *_sd3.npz files have been removed from $DIR and its subdirectories."