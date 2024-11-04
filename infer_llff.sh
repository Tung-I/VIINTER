# Define variables
DATASET="llff"
BASE_DIR="$HOME/datasets/$DATASET"
EXP_DIR="exps"
FRAME="0000"
MODEL_CHECKPOINT="200000_clip_0.01_dim128_W512_D8/net_20000.pth"

# Define categories and camera pairs
CATEGORIES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
CAMERA_PAIRS=("cam03_cam05" "cam09_cam11")

# Loop through each category and camera pair
for CATEGORY in "${CATEGORIES[@]}"; do
    for CAM_PAIR in "${CAMERA_PAIRS[@]}"; do
        MODEL_PATH="${EXP_DIR}/${DATASET}_${CATEGORY}_${CAM_PAIR}_${FRAME}/${MODEL_CHECKPOINT}"
        DATA_DIR="${BASE_DIR}/${CATEGORY}/view_pair/${CAM_PAIR}_${FRAME}/"
        
        # Run the evaluation command
        python infer.py --model_path "$MODEL_PATH" --data_dir "$DATA_DIR" --dataset "$DATASET"
    done
done