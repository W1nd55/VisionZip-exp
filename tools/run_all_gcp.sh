#!/usr/bin/env bash
# Helper script to run evaluations for multiple architectures on GCP or local GPU nodes.
# Usage: edit the variables below or call with env vars set, then run:
#    bash tools/run_all_gcp.sh

set -euo pipefail

# ----------------- Configuration (EDIT ME) -----------------
# Path to repo root (auto-detected)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Data roots - set these to where you've downloaded datasets
MME_ROOT="${MME_ROOT:-$REPO_ROOT/datasets/mme/MME_Benchmark_release_version/MME_Benchmark}"
POPE_ROOT="${POPE_ROOT:-$REPO_ROOT/datasets/pope}"
COCO_ROOT="${COCO_ROOT:-$REPO_ROOT/datasets/coco}"
DOCVQA_ROOT="${DOCVQA_ROOT:-$REPO_ROOT/datasets/docvqa}"

# Models (HF IDs or local paths)
LLAVA_MODEL="${LLAVA_MODEL:-liuhaotian/llava-v1.5-7b}"
VISIONZIP_MODEL="${VISIONZIP_MODEL:-liuhaotian/llava-v1.5-7b}"
SPARSEVLM_MODEL="${SPARSEVLM_MODEL:-liuhaotian/llava-v1.5-7b}"

# Output root where per-dataset results will be stored
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/eval_results_gcp}"

# Python executable (virtualenv / conda activate externally)
PYTHON=${PYTHON:-python3}

# Extra flags (e.g., --limit 100 for smoke tests)
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# ----------------- Runner -----------------
echo "Repo root: $REPO_ROOT"
mkdir -p "$OUT_ROOT"

cd "$REPO_ROOT"

echo "Running MME evaluations for architectures: Llava, VisionZip, SparseVLM, SparseZip"

mkdir -p "$OUT_ROOT/mme"

echo "-> Llava (baseline)"
$PYTHON tools/mme_run_all.py --cfg config/llava_mme.yaml --mme_root "$MME_ROOT" --out_root "$OUT_ROOT/mme/llava" --model_type llava_vzip --model_path "$LLAVA_MODEL" $EXTRA_FLAGS

echo "-> VisionZip"
$PYTHON tools/mme_run_all.py --cfg config/visionzip_mme.yaml --mme_root "$MME_ROOT" --out_root "$OUT_ROOT/mme/visionzip" --model_type llava_vzip --model_path "$VISIONZIP_MODEL" $EXTRA_FLAGS

echo "-> SparseVLM"
$PYTHON tools/mme_run_all.py --cfg config/sparsevlm_mme.yaml --mme_root "$MME_ROOT" --out_root "$OUT_ROOT/mme/sparsevlm" --model_type sparsevlm --model_path "$SPARSEVLM_MODEL" $EXTRA_FLAGS

echo "-> SparseZip"
$PYTHON tools/mme_run_all.py --cfg config/sparsezip_mme.yaml --mme_root "$MME_ROOT" --out_root "$OUT_ROOT/mme/sparsezip" --model_type sparsezip --model_path "$SPARSEVLM_MODEL" $EXTRA_FLAGS

echo "Running POPE evaluations"
mkdir -p "$OUT_ROOT/pope"

echo "-> Llava (baseline)"
$PYTHON tools/pope_run_all.py --pope_root "$POPE_ROOT" --img_root "$COCO_ROOT/val2014" --out_root "$OUT_ROOT/pope/llava" --cfg config/llava_pope.yaml --model_type llava_vzip --model_path "$LLAVA_MODEL" $EXTRA_FLAGS

echo "-> VisionZip"
$PYTHON tools/pope_run_all.py --pope_root "$POPE_ROOT" --img_root "$COCO_ROOT/val2014" --out_root "$OUT_ROOT/pope/visionzip" --cfg config/visionzip_pope.yaml --model_type llava_vzip --model_path "$VISIONZIP_MODEL" $EXTRA_FLAGS

echo "-> SparseVLM"
$PYTHON tools/pope_run_all.py --pope_root "$POPE_ROOT" --img_root "$COCO_ROOT/val2014" --out_root "$OUT_ROOT/pope/sparsevlm" --cfg config/sparsevlm_pope.yaml --model_type sparsevlm --model_path "$SPARSEVLM_MODEL" $EXTRA_FLAGS

echo "-> SparseZip"
$PYTHON tools/pope_run_all.py --pope_root "$POPE_ROOT" --img_root "$COCO_ROOT/val2014" --out_root "$OUT_ROOT/pope/sparsezip" --cfg config/sparsezip_pope.yaml --model_type sparsezip --model_path "$SPARSEVLM_MODEL" $EXTRA_FLAGS

echo "Running DocVQA evaluations"
mkdir -p "$OUT_ROOT/docvqa"

echo "-> Llava"
$PYTHON tools/docvqa_run_all.py --docvqa_root "$DOCVQA_ROOT" --out_root "$OUT_ROOT/docvqa/llava" --cfg config/llava_docvqa.yaml --model_type llava_vzip --model_path "$LLAVA_MODEL" $EXTRA_FLAGS

echo "-> VisionZip"
$PYTHON tools/docvqa_run_all.py --docvqa_root "$DOCVQA_ROOT" --out_root "$OUT_ROOT/docvqa/visionzip" --cfg config/visionzip_docvqa.yaml --model_type llava_vzip --model_path "$VISIONZIP_MODEL" $EXTRA_FLAGS

echo "-> SparseVLM"
$PYTHON tools/docvqa_run_all.py --docvqa_root "$DOCVQA_ROOT" --out_root "$OUT_ROOT/docvqa/sparsevlm" --cfg config/sparsevlm_docvqa.yaml --model_type sparsevlm --model_path "$SPARSEVLM_MODEL" $EXTRA_FLAGS

echo "-> SparseZip"
$PYTHON tools/docvqa_run_all.py --docvqa_root "$DOCVQA_ROOT" --out_root "$OUT_ROOT/docvqa/sparsezip" --cfg config/sparsezip_docvqa.yaml --model_type sparsezip --model_path "$SPARSEVLM_MODEL" $EXTRA_FLAGS

echo "Running COCO Caption evaluations (optional)"
mkdir -p "$OUT_ROOT/coco"

echo "-> Llava"
$PYTHON tools/coco_cap_run_all.py --ann_path "$COCO_ROOT/annotations/captions_val2014.json" --out_root "$OUT_ROOT/coco/llava" --cfg config/llava_coco_cap.yaml --model_type llava_vzip --model_path "$LLAVA_MODEL" $EXTRA_FLAGS

echo "-> VisionZip"
$PYTHON tools/coco_cap_run_all.py --ann_path "$COCO_ROOT/annotations/captions_val2014.json" --out_root "$OUT_ROOT/coco/visionzip" --cfg config/visionzip_coco_cap.yaml --model_type llava_vzip --model_path "$VISIONZIP_MODEL" $EXTRA_FLAGS

echo "-> SparseVLM"
$PYTHON tools/coco_cap_run_all.py --ann_path "$COCO_ROOT/annotations/captions_val2014.json" --out_root "$OUT_ROOT/coco/sparsevlm" --cfg config/sparsevlm_docvqa.yaml --model_type sparsevlm --model_path "$SPARSEVLM_MODEL" $EXTRA_FLAGS

echo "-> SparseZip"
$PYTHON tools/coco_cap_run_all.py --ann_path "$COCO_ROOT/annotations/captions_val2014.json" --out_root "$OUT_ROOT/coco/sparsezip" --cfg config/sparsezip_mme.yaml --model_type sparsezip --model_path "$SPARSEVLM_MODEL" $EXTRA_FLAGS

echo "All runs finished. Aggregated outputs are under $OUT_ROOT"
