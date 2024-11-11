#!/bin/bash

# Check if data directory is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <data_dir>"
    exit 1
fi

# Set the dataset directory from the first argument
DATA_DIR=$1



python train.py --data_dir "$DATA_DIR" --dset dynerf --category sear_steak --scene cam19_cam20_0050 --clip 0.01 --exp_name 200000 --iters 200000
python train.py --data_dir "$DATA_DIR" --dset mipnerf360 --category room --scene cam15_cam16_0000 --clip 0.01 --exp_name 200000 --iters 200000
