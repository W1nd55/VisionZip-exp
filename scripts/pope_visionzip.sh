#!/bin/bash

# VisionZip Evaluation Script for POPE Dataset
# Simple wrapper that calls the Python evaluation script

# Configuration
MODEL_PATH="liuhaotian/llava-v1.5-7b"  # Use 7B for laptop GPU
VISIONZIP_DOMINANT=54
VISIONZIP_CONTEXTUAL=10
GPU_ID=0

echo "=========================================="
echo "VisionZip POPE Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "VisionZip: dominant=$VISIONZIP_DOMINANT, contextual=$VISIONZIP_CONTEXTUAL"
echo "GPU: $GPU_ID"
echo "=========================================="

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run Python evaluation script
python scripts/eval_pope_visionzip.py \
    --model-path "$MODEL_PATH" \
    --dominant $VISIONZIP_DOMINANT \
    --contextual $VISIONZIP_CONTEXTUAL \
    --image-folder "./playground/eval/pope/val2014" \
    --question-file "./playground/eval/pope/llava_pope_test.jsonl" \
    --answers-file "./playground/eval/pope/answers/visionzip-v1.5-7b.jsonl" \
    --temperature 0 \
    --num-beams 1

