#!/bin/bash

# Define dataset details
data_dir="/home/ubuntu/dataset_for_A100"
dataset="llff"
configs=('none')
exp_names=('none')
frame_id=0

# Categories and camera pairs
CATEGORIES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
declare -A CAMERA_PAIRS=( ["03"]="04" ["09"]="10" ["03"]="05" ["09"]="11" )
# declare -A CAMERA_PAIRS=( ["03"]="04" ["09"]="10" )


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