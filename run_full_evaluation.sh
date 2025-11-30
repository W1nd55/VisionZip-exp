#!/bin/bash
set -e

echo "🚀 Starting Full Benchmark Evaluation"
echo "======================================"
echo ""

# Set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/models/LLaVA:$(pwd)/models/VisionZip:$(pwd)

# Output directory
RESULTS_DIR="eval_results_final"
mkdir -p "$RESULTS_DIR"

# Dataset paths
POPE_ROOT="datasets/pope"
COCO_ROOT="datasets/coco/val2014"
MME_ROOT="datasets/mme/MME_Benchmark_release_version/MME_Benchmark"

echo "📊 Running evaluations for 4 architectures x 2 datasets = 8 runs"
echo ""

# ============================================
# POPE Evaluations
# ============================================

echo "🔍 [1/8] Running POPE - SparseZip (Optimized)"
python tools/pope_run_all.py \
    --pope_root "$POPE_ROOT" \
    --img_root "$COCO_ROOT" \
    --out_root "$RESULTS_DIR/pope_sparsezip" \
    --cfg config/sparsezip_pope.yaml

echo "🔍 [2/8] Running POPE - VisionZip"
python tools/pope_run_all.py \
    --pope_root "$POPE_ROOT" \
    --img_root "$COCO_ROOT" \
    --out_root "$RESULTS_DIR/pope_visionzip" \
    --cfg config/visionzip_pope.yaml

echo "🔍 [3/8] Running POPE - SparseVLM"
python tools/pope_run_all.py \
    --pope_root "$POPE_ROOT" \
    --img_root "$COCO_ROOT" \
    --out_root "$RESULTS_DIR/pope_sparsevlm" \
    --cfg config/sparsevlm_pope.yaml

echo "🔍 [4/8] Running POPE - Llava (Baseline)"
python tools/pope_run_all.py \
    --pope_root "$POPE_ROOT" \
    --img_root "$COCO_ROOT" \
    --out_root "$RESULTS_DIR/pope_llava" \
    --cfg config/llava_pope.yaml

# ============================================
# MME Evaluations
# ============================================

echo "🔍 [5/8] Running MME - SparseZip (Optimized)"
python tools/mme_run_all.py \
    --mme_root "$MME_ROOT" \
    --out_root "$RESULTS_DIR/mme_sparsezip" \
    --cfg config/sparsezip_mme.yaml

echo "🔍 [6/8] Running MME - VisionZip"
python tools/mme_run_all.py \
    --mme_root "$MME_ROOT" \
    --out_root "$RESULTS_DIR/mme_visionzip" \
    --cfg config/visionzip_mme.yaml

echo "🔍 [7/8] Running MME - SparseVLM"
python tools/mme_run_all.py \
    --mme_root "$MME_ROOT" \
    --out_root "$RESULTS_DIR/mme_sparsevlm" \
    --cfg config/sparsevlm_mme.yaml

echo "🔍 [8/8] Running MME - Llava (Baseline)"
python tools/mme_run_all.py \
    --mme_root "$MME_ROOT" \
    --out_root "$RESULTS_DIR/mme_llava" \
    --cfg config/llava_mme.yaml

echo ""
echo "✅ All evaluations complete!"
echo ""
echo "📈 Aggregating results into summary.csv..."

# Create summary CSV aggregator
python - << 'PYTHON_SCRIPT'
import csv
import os
import json
from pathlib import Path

results_dir = Path("eval_results_final")

# Initialize results
results = {}

def parse_pope_results(model_name, base_path):
    """Parse POPE results from summary.csv files"""
    accuracy_sum = 0.0
    latency_sum = 0.0
    count = 0
    
    for variant in ['random', 'popular', 'adversarial']:
        summary_file = base_path / f"results/outputs_{variant}/summary.csv"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                reader = csv.reader(f)
                metrics = dict(reader)
                
                # Get accuracy and latency
                acc_key = next((k for k in metrics.keys() if 'acc' in k.lower()), None)
                if acc_key:
                    accuracy_sum += float(metrics[acc_key])
                    
                lat_key = next((k for k in metrics.keys() if 'end2end_ms' in k.lower() and 'avg' in k.lower()), None)
                if lat_key:
                    latency_sum += float(metrics[lat_key])
                    
                count += 1
    
    if count > 0:
        return accuracy_sum / count, latency_sum / count
    return None, None

def parse_mme_results(model_name, base_path):
    """Parse MME results from summary.csv"""
    summary_file = base_path / "summary.csv"
    if not summary_file.exists():
        return None, None
        
    with open(summary_file, 'r') as f:
        reader = csv.reader(f)
        metrics = dict(reader)
        
        # Get overall accuracy and latency
        acc_key = next((k for k in metrics.keys() if 'mme_acc' in k.lower() and 'overall' in k.lower()), None)
        if not acc_key:
            acc_key = next((k for k in metrics.keys() if 'mme_acc' in k.lower()), None)
            
        lat_key = next((k for k in metrics.keys() if 'end2end_ms' in k.lower() and 'avg' in k.lower()), None)
        
        accuracy = float(metrics[acc_key]) if acc_key and acc_key in metrics else None
        latency = float(metrics[lat_key]) if lat_key and lat_key in metrics else None
        
        return accuracy, latency

# Parse results for each model
models = [
    ('SparseZip', 'sparsezip'),
    ('VisionZip', 'visionzip'),
    ('SparseVLM', 'sparsevlm'),
    ('Llava', 'llava'),
]

for model_name, model_key in models:
    results[model_name] = {}
    
    # POPE
    pope_acc, pope_lat = parse_pope_results(model_name, results_dir / f"pope_{model_key}")
    if pope_acc is not None:
        results[model_name]['pope_acc'] = f"{pope_acc:.3f}"
        results[model_name]['pope_latency'] = f"{int(pope_lat)}"
    
    # MME
    mme_acc, mme_lat = parse_mme_results(model_name, results_dir / f"mme_{model_key}")
    if mme_acc is not None:
        results[model_name]['mme_acc'] = f"{mme_acc:.3f}"
        results[model_name]['mme_latency'] = f"{int(mme_lat)}"

# Write to CSV
output_file = "summary_final.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Header
    writer.writerow([
        'Model',
        'POPE Accuracy', 'POPE Latency (ms)',
        'MME Accuracy', 'MME Latency (ms)'
    ])
    
    # Rows
    for model_name in ['Llava', 'VisionZip', 'SparseVLM', 'SparseZip']:
        row = [model_name]
        r = results[model_name]
        row.extend([
            r.get('pope_acc', 'N/A'),
            r.get('pope_latency', 'N/A'),
            r.get('mme_acc', 'N/A'),
            r.get('mme_latency', 'N/A'),
        ])
        writer.writerow(row)

print(f"✅ Summary saved to {output_file}")

# Display results
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
with open(output_file, 'r') as f:
    print(f.read())
PYTHON_SCRIPT

echo ""
echo "🎉 Evaluation complete! Check summary_final.csv for results."
