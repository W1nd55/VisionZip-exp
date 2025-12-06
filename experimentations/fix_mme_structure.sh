#!/bin/bash

# Current configured path
CURRENT_MME="/home/w1nd519994824/project/mittalshivam003/VisionZip-exp/datasets/mme/MME_Benchmark_release_version"

echo "🔍 Checking MME structure at: $CURRENT_MME"

# Check if there is a nested 'MME_Benchmark' folder
if [ -d "$CURRENT_MME/MME_Benchmark" ]; then
    echo "⚠️ Found nested 'MME_Benchmark' folder. Adjusting path..."
    NEW_MME="$CURRENT_MME/MME_Benchmark"
elif [ -d "$CURRENT_MME/MME_Benchmark_release_version" ]; then
    echo "⚠️ Found nested 'MME_Benchmark_release_version' folder. Adjusting path..."
    NEW_MME="$CURRENT_MME/MME_Benchmark_release_version"
else
    echo "ℹ️ No obvious nesting found. Listing directory content to debug:"
    ls -F "$CURRENT_MME"
    NEW_MME="$CURRENT_MME"
fi

echo "✅ Setting MME Root to: $NEW_MME"

# Rewrite run_mme_eval.sh with the NEW path
cat > run_mme_eval.sh <<EOF
#!/bin/bash
MME_ROOT="$NEW_MME"
RESULTS_DIR="results"
MODEL_ROOT="/home/w1nd519994824/project/mittalshivam003/models"
BASE_DIR=\$(pwd)

mkdir -p "\$RESULTS_DIR"

if [ ! -d "\$MME_ROOT" ]; then
    echo "❌ Error: MME_ROOT (\$MME_ROOT) does not exist."
    exit 1
fi

# Assume conda env is active
export PYTHONPATH="\$BASE_DIR:\$PYTHONPATH"

echo "🚀 Running MME Eval with MME_ROOT=\$MME_ROOT"

# 1. Llava
echo "Running Llava..."
python tools/mme_run_all.py --mme_root "\$MME_ROOT" --out_root "\$RESULTS_DIR/mme_llava" --model_path "\$MODEL_ROOT/llava-v1.5-7b" --cfg config/default.yaml

# 2. VisionZip
echo "Running VisionZip..."
python tools/mme_run_all.py --mme_root "\$MME_ROOT" --out_root "\$RESULTS_DIR/mme_visionzip" --model_path "\$MODEL_ROOT/llava-v1.5-7b" --cfg config/default.yaml --dominant 54 --contextual 10

# 3. SparseVLM
echo "Running SparseVLM..."
python tools/mme_run_all.py --mme_root "\$MME_ROOT" --out_root "\$RESULTS_DIR/mme_sparsevlm" --model_path "\$MODEL_ROOT/SparseVLM_7B_224" --model_type sparsevlm --cfg config/default.yaml

# 4. SparseZip
echo "Running SparseZip..."
python tools/mme_run_all.py --mme_root "\$MME_ROOT" --out_root "\$RESULTS_DIR/mme_sparsezip" --cfg config/sparsezip_mme.yaml

# Aggregate
python3 -c '
import os, csv, glob
results_dir = "results"
output_csv = os.path.join(results_dir, "summary_final.csv")
models = [
    {"name": "Llava", "dir": "mme_llava"},
    {"name": "VisionZip", "dir": "mme_visionzip"},
    {"name": "SparseVLM", "dir": "mme_sparsevlm"},
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
EOF
chmod +x run_mme_eval.sh
echo "✅ Updated run_mme_eval.sh"

# Run Evaluation
echo "🏃 Starting Evaluation..."
./run_mme_eval.sh
