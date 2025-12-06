#!/bin/bash

BASE_CONFIG="config/sparsezip_pope.yaml"

POPE_ROOT="/home/w1nd519994824/project/VisionZip-exp/datasets/pope"
IMG_ROOT="/home/w1nd519994824/project/VisionZip-exp/datasets/coco/val2014"
OUT_ROOT_BASE="/home/w1nd519994824/project/VisionZip-exp/eval_results/pope_eval_results"

mkdir -p "$OUT_ROOT_BASE"

run_ablation() {
    TARGET_FLAG=$1 
    
    echo "========================================================"
    echo "Starting Ablation Task: ${TARGET_FLAG} = True"
    echo "========================================================"

    TEMP_CONFIG="config/temp_${TARGET_FLAG}.yaml"
    
    cp "$BASE_CONFIG" "$TEMP_CONFIG"
    
    sed -i "s/${TARGET_FLAG}: false/${TARGET_FLAG}: true/g" "$TEMP_CONFIG"
    TASK_OUT_DIR="${OUT_ROOT_BASE}/pope_eval_results_sparsezip_${TARGET_FLAG}"
    LOG_FILE="${OUT_ROOT_BASE}/log_sparsezip_${TARGET_FLAG}.txt"
    
    echo "Config: ${TEMP_CONFIG}"
    echo "Output Dir: ${TASK_OUT_DIR}"
    echo "Logging to: ${LOG_FILE}"

    python tools/pope_run_all.py \
        --pope_root "$POPE_ROOT" \
        --img_root "$IMG_ROOT" \
        --out_root "$TASK_OUT_DIR" \
        --cfg "$TEMP_CONFIG" > "$LOG_FILE" 2>&1

    rm "$TEMP_CONFIG"
    
    echo "Finished Task: ${TARGET_FLAG}"
    echo ""
}

run_ablation "skip_hybrid_attn"
run_ablation "skip_dynamic_k"
run_ablation "skip_ctx_merge"

echo "All ablation studies completed."