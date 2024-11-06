# # Define variables
# DATASET="mipnerf360"
# BASE_DIR="$HOME/datasets/$DATASET"
# EXP_DIR="exps"
# FRAME="0000"
# MODEL_CHECKPOINT="200000_clip_0.01_dim128_W512_D8/net_60000.pth"

# # Define categories and camera pairs
# CATEGORIES=("bicycle" "counsai" "counter" "kitchen" "room")
# CAMERA_PAIRS=("cam03_cam05" "cam09_cam11")

# # Loop through each category and camera pair
# for CATEGORY in "${CATEGORIES[@]}"; do
#     for CAM_PAIR in "${CAMERA_PAIRS[@]}"; do
#         MODEL_PATH="${EXP_DIR}/${DATASET}_${CATEGORY}_${CAM_PAIR}_${FRAME}/${MODEL_CHECKPOINT}"
#         DATA_DIR="${BASE_DIR}/${CATEGORY}/view_pair/${CAM_PAIR}_${FRAME}/"
        
#         # Run the evaluation command
#         python infer.py --model_path "$MODEL_PATH" --data_dir "$DATA_DIR" --dataset "$DATASET"
#     done
# done
# python infer.py --model_path exps/mipnerf360_bonsai_cam11_cam12_0000/200000_clip_0.01_dim128_W512_D8/net_60000.pth --data_dir ~/datasets/mipnerf360/bonsai/view_pair/cam11_cam12_0000/ --dataset mipnerf360
# python infer.py --model_path exps/mipnerf360_counter_cam13_cam14_0000/200000_clip_0.01_dim128_W512_D8/net_90000.pth --data_dir ~/datasets/mipnerf360/counter/view_pair/cam13_cam14_0000/ --dataset mipnerf360
# python infer.py --model_path exps/mipnerf360_kitchen_cam13_cam14_0000/200000_clip_0.01_dim128_W512_D8/net_60000.pth --data_dir ~/datasets/mipnerf360/kitchen/view_pair/cam13_cam14_0000/ --dataset mipnerf360
# python infer.py --model_path exps/mipnerf360_room_cam11_cam12_0000/200000_clip_0.01_dim128_W512_D8/net_90000.pth --data_dir ~/datasets/mipnerf360/room/view_pair/cam11_cam12_0000/ --dataset mipnerf360

# python infer.py --model_path exps/dynerf_coffee_martini_cam08_cam09_0050/200000_clip_0.01_dim128_W512_D8/net_50000.pth --data_dir ~/datasets/dynerf/coffee_martini/view_pair/cam08_cam09_0050/ --dataset dynerf
# python infer.py --model_path exps/dynerf_cook_spinach_cam17_cam18_0050/200000_clip_0.01_dim128_W512_D8/net_50000.pth --data_dir ~/datasets/dynerf/cook_spinach/view_pair/cam17_cam18_0050/ --dataset dynerf
# python infer.py --model_path exps/dynerf_flame_salmon_1_cam01_cam02_0050/200000_clip_0.01_dim128_W512_D8/net_50000.pth --data_dir ~/datasets/dynerf/flame_salmon_1/view_pair/cam01_cam02_0050/ --dataset dynerf


# python infer.py --model_path exps/llff_leaves_cam09_cam10_0000/200000_clip_0.01_dim128_W512_D8/net_40000.pth --data_dir ~/datasets/llff/leaves/view_pair/cam09_cam10_0000/ --dataset llff
# python infer.py --model_path exps/llff_fortress_cam02_cam03_0000/200000_clip_0.01_dim128_W512_D8/net_60000.pth --data_dir ~/datasets/llff/fortress/view_pair/cam02_cam03_0000/ --dataset llff
