#!/bin/bash
MME_ROOT="/home/w1nd519994824/project/mittalshivam003/VisionZip-exp/datasets/mme/MME_Benchmark_release_version/MME_Benchmark"
RESULTS_DIR="results"
BASE_DIR="/home/w1nd519994824/project/mittalshivam003/VisionZip-exp"

# Set HF Home to avoid boot disk full
export HF_HOME="/home/w1nd519994824/project/mittalshivam003/VisionZip-exp/hf_cache"
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

echo "🚀 Running MME Eval..."

# 1. Llava (Download from HF)
echo "🔍 [1/3] Running Llava (Baseline)"
python tools/mme_run_all.py --mme_root "$MME_ROOT" --out_root "$RESULTS_DIR/mme_llava" --model_path "liuhaotian/llava-v1.5-7b" --cfg config/default.yaml

# 2. VisionZip (Download from HF)
echo "🔍 [2/3] Running VisionZip"
python tools/mme_run_all.py --mme_root "$MME_ROOT" --out_root "$RESULTS_DIR/mme_visionzip" --model_path "liuhaotian/llava-v1.5-7b" --cfg config/default.yaml --dominant 54 --contextual 10

# 3. SparseZip (Download from HF)
echo "🔍 [3/3] Running SparseZip"
python tools/mme_run_all.py --mme_root "$MME_ROOT" --out_root "$RESULTS_DIR/mme_sparsezip" --model_path "liuhaotian/llava-v1.5-7b" --cfg config/sparsezip_mme.yaml

# Aggregate
python3 -c '
import os, csv
results_dir = "results"
output_csv = os.path.join(results_dir, "summary_final.csv")
models = [
    {"name": "Llava", "dir": "mme_llava"},
    {"name": "VisionZip", "dir": "mme_visionzip"},
    {"name": "SparseZip", "dir": "mme_sparsezip"},
]
final = []
for m in models:
    p = os.path.join(results_dir, m["dir"], "mme_summary.csv")
    if os.path.exists(p):
        with open(p) as f:
            for row in csv.DictReader(f):
                row["Model"] = m["name"]
                final.append(row)
if final:
    keys = ["Model"] + sorted([k for k in final[0].keys() if k != "Model"])
    with open(output_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(final)
    print(f"✅ Saved {output_csv}")
else:
    print("❌ No results to aggregate")
'
