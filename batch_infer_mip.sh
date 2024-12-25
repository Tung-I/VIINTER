#!/bin/bash

# Define dataset details
data_dir="/home/ubuntu/dataset_for_A100"
dataset="mipnerf360"
configs=(none)
exp_names=('none')
frame_id=0

# Categories and camera pairs
CATEGORIES=("bicycle" "bonsai" "counter" "garden" "kitchen" "room")
declare -A CAMERA_PAIRS=( ["01"]="02" ["02"]="03" ["03"]="04" ["11"]="12" ["12"]="13" ["13"]="14" ["15"]="16" )


# cd ..

# Loop through each configuration, category, and camera pair
for config_idx in "${!configs[@]}"; do
  config="${configs[$config_idx]}"
  exp_name="${exp_names[$config_idx]}"
  
  for category in "${CATEGORIES[@]}"; do
    for im0_id in "${!CAMERA_PAIRS[@]}"; do
      im1_id="${CAMERA_PAIRS[$im0_id]}"
      
      python infer.py  --ckpt_dir "/home/ubuntu/exps_done" \
                      --data_dir "$data_dir" \
                      --save_frames \
                      --dataset "$dataset" \
                      --category "$category" \
                      --im0_id "$im0_id" \
                      --im1_id "$im1_id" \
                      --iter 200000\
                      --frame_id "$frame_id" 

      python infer.py  --ckpt_dir "/home/ubuntu/exps_done" \
                      --data_dir "$data_dir" \
                      --save_frames \
                      --dataset "$dataset" \
                      --category "$category" \
                      --im0_id "$im0_id" \
                      --im1_id "$im1_id" \
                      --iter 100000\
                      --frame_id "$frame_id" 
                      
    done
  done
done