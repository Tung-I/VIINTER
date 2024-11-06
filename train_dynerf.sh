#!/bin/bash

# Check if data directory is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <data_dir>"
    exit 1
fi

# Set the dataset directory from the first argument
DATA_DIR=$1

# List of categories and scenes
categories=("coffee_martini" "cook_spinach" "cut_roasted_beef" "flame_salmon_1" "flame_steak" "sear_steak")
scenes=("cam08_cam09_0050" "cam11_cam12_0050" "cam18_cam19_0050" "cam19_cam20_0050")

# Outer loop over categories
for category in "${categories[@]}"; do
    # Inner loop over scenes
    for scene in "${scenes[@]}"; do
        python train.py --data_dir "$DATA_DIR" --dset dynerf --category "$category" --scene "$scene" --clip 0.01 --exp_name 200000 --iters 200000
    done
done
