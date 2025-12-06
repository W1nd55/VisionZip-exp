#!/bin/bash
# run_cross_attention_eval.sh
# Runs full MME and POPE evaluation for SparseZip with cross-attention enabled
# Results saved to ./results_cross_attention/

set -e  # Exit on error

echo "========================================"
echo "SparseZip Cross-Attention Evaluation"
echo "========================================"
echo ""

# Configuration
OUT_ROOT="./results_cross_attention"
MME_ROOT="datasets/mme/MME_Benchmark_release_version/MME_Benchmark"
POPE_ROOT="datasets/pope"
IMG_ROOT="datasets/coco/val2014"
MODEL_PATH="liuhaotian/llava-v1.5-7b"

# Create output directory
mkdir -p "$OUT_ROOT"

echo "Output directory: $OUT_ROOT"
echo ""

# ==================== MME Evaluation ====================
echo "=========================================="
echo "1. Running MME Evaluation"
echo "=========================================="
echo "Config: config/sparsezip_mme.yaml"
echo "Output: $OUT_ROOT/mme_sparsezip"
echo ""

python tools/mme_run_all.py \
  --mme_root "$MME_ROOT" \
  --out_root "$OUT_ROOT/mme_sparsezip" \
  --cfg config/sparsezip_mme.yaml \
  --model_path "$MODEL_PATH"

echo ""
echo "✓ MME evaluation complete!"
echo "  Results: $OUT_ROOT/mme_sparsezip/mme_summary.csv"
echo ""

# ==================== POPE Evaluation ====================
echo "=========================================="
echo "2. Running POPE Evaluation"
echo "=========================================="
echo "Config: config/sparsezip_pope.yaml"
echo "Output: $OUT_ROOT/pope_sparsezip"
echo ""

export PYTHONPATH=$PYTHONPATH:.
python tools/pope_run_all.py \
  --pope_root "$POPE_ROOT" \
  --img_root "$IMG_ROOT" \
  --out_root "$OUT_ROOT/pope_sparsezip" \
  --cfg config/sparsezip_pope.yaml \
  --model_path "$MODEL_PATH"

echo ""
echo "✓ POPE evaluation complete!"
echo "  Results: $OUT_ROOT/pope_sparsezip/pope_summary.csv"
echo ""

# ==================== Summary ====================
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUT_ROOT/"
echo ""
echo "To view results:"
echo "  MME:  cat $OUT_ROOT/mme_sparsezip/mme_summary.csv"
echo "  POPE: cat $OUT_ROOT/pope_sparsezip/pope_summary.csv"
echo ""
