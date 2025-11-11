#!/bin/bash
#
# Run MME evaluations for VisionZip and SparseVLM
#

set -e  # Exit on error

WORKSPACE="/u/m/a/maheenabooba/cs769"
cd "$WORKSPACE"
source venv/bin/activate

echo "============================================================"
echo "MME EVALUATION - VISIONZIP AND SPARSEVLM"
echo "============================================================"
echo ""

# Set number of samples (use -1 for full evaluation, or small number for testing)
NUM_SAMPLES=${1:-10}  # Default to 10 for testing

if [ "$NUM_SAMPLES" -eq -1 ]; then
    echo "Running FULL evaluation on all 2374 samples"
    echo "⚠ This will take several hours!"
else
    echo "Running TEST evaluation on $NUM_SAMPLES samples"
fi
echo ""

# Run VisionZip evaluation
echo "============================================================"
echo "1. VISIONZIP EVALUATION"
echo "============================================================"
python scripts/eval_mme_visionzip.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --dominant-tokens 54 \
    --contextual-tokens 10 \
    --clip-layer 22 \
    --num-samples $NUM_SAMPLES \
    --experiment-name visionzip-v1.5-7b

echo ""
echo "============================================================"
echo "2. SPARSEVLM EVALUATION"
echo "============================================================"
python scripts/eval_mme_sparsevlm.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --retained-tokens 64 \
    --sparse-layers "2,6,15" \
    --token-counts "66,30,17" \
    --num-samples $NUM_SAMPLES \
    --experiment-name sparsevlm-v1.5-7b

echo ""
echo "============================================================"
echo "3. COMPARISON"
echo "============================================================"
echo "VisionZip results:"
cat models/LLaVA/playground/eval/MME/eval_tool/answers/visionzip-v1.5-7b/*.txt | head -5
echo ""
echo "SparseVLM results:"
cat models/LLaVA/playground/eval/MME/eval_tool/answers/sparsevlm-v1.5-7b/*.txt | head -5

echo ""
echo "============================================================"
echo "✓ MME EVALUATIONS COMPLETE!"
echo "============================================================"
echo "Results saved in:"
echo "  - models/LLaVA/playground/eval/MME/answers/visionzip-v1.5-7b.jsonl"
echo "  - models/LLaVA/playground/eval/MME/answers/sparsevlm-v1.5-7b.jsonl"
echo "============================================================"

