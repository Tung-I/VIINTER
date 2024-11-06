#!/bin/bash

# Check if data directory is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <data_dir>"
    exit 1
fi

# Set the dataset directory from the first argument
DATA_DIR=$1

# List of categories and scenes
categories=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
scenes=("cam03_cam04_0000" "cam03_cam05_0000" "cam09_cam10_0000" "cam09_cam11_0000")

# Outer loop over categories
for category in "${categories[@]}"; do
    # Inner loop over scenes
    for scene in "${scenes[@]}"; do
        python train.py --data_dir "$DATA_DIR" --dset llff --category "$category" --scene "$scene" --clip 0.01 --exp_name 200000 --iters 200000
    done
done
