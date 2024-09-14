#!/bin/bash

# Define base data directory and dataset details
base_data_dir="/home/tungi/datasets/dynerf"
model_base_path="exps/dynerf"
frame_id="0050"
dataset="dynerf"

# Array of subclasses
subclasses=("coffee_martini" "cook_spinach" "cut_roasted_beef" "flame_steak" "sear_steak")

# Loop through each subclass
for subclass in "${subclasses[@]}"; do
    echo "Processing subclass: $subclass"
    
    # Loop through camera pairs from cam01_cam02 to cam19_cam20
    for (( i=1; i<=19; i++ )); do
        cam_current=$(printf "%02d" $((i-1)))  # Format numbers to two digits
        cam_next=$(printf "%02d" $i)
        cam_pair="cam${cam_current}_cam${cam_next}_${frame_id}"
        model_path="${model_base_path}_${subclass}_${cam_pair}/exp_clip_0.01_dim128_W512_D8/net.pth"
        view_pair_dir="${base_data_dir}/${subclass}/view_pair/${cam_pair}"
        
        # Check if the view pair directory exists
        if [ -d "$view_pair_dir" ]; then
            # Build and run the command
            command="python infer.py --model_path $model_path --data_dir $view_pair_dir --dataset $dataset"
            echo "Running: $command"
            eval $command
        else
            echo "Skipping: $view_pair_dir does not exist."
        fi
    done
done
