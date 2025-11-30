#!/bin/bash

# 1. Define Paths
BASE_DIR=$(pwd)
MME_DIR="/home/w1nd519994824/project/mittalshivam003/VisionZip-exp/datasets/mme/MME_Benchmark_release_version/MME_Benchmark"
RESULTS_DIR="results"
HF_CACHE="$BASE_DIR/hf_cache"

# Ensure MME exists
if [ ! -d "$MME_DIR" ]; then
    # Fallback to the parent if nested one doesn't exist
    MME_DIR="/home/w1nd519994824/project/mittalshivam003/VisionZip-exp/datasets/mme/MME_Benchmark_release_version"
fi
if [ ! -d "$MME_DIR" ]; then
    echo "❌ Error: Could not find MME directory at $MME_DIR"
    exit 1
fi

echo "✅ Using MME Root: $MME_DIR"
echo "✅ Using HF Cache: $HF_CACHE"
mkdir -p "$HF_CACHE"
mkdir -p "$RESULTS_DIR"

# 2. Re-apply Syntax Fix (Safety First)
echo "🔧 Applying Syntax Fix to modelling_sparse_llama.py..."
python3 -c '
import os
path = "models/SparseVLMs/llava/model/language_model/modelling_sparse_llama.py"
if os.path.exists(path):
    with open(path, "r") as f: content = f.read()
    lines = content.splitlines()
    new_lines = []
    skip = False
    fixed = False
    for line in lines:
        if "if eos_token_id_tensor is not None:" in line:
            new_lines.append(line)
            indent = line[:line.find("if")] + "    "
            new_lines.append(indent + "# Simplified EOS check (Fixed)")
            new_lines.append(indent + "is_eos = torch.isin(next_tokens, eos_token_id_tensor)")
            new_lines.append(indent + "unfinished_sequences = unfinished_sequences.mul((~is_eos).long())")
            skip = True
            fixed = True
        elif skip:
            if "if unfinished_sequences.max() == 0:" in line:
                new_lines.append(line)
                skip = False
        else:
            new_lines.append(line)
    if fixed:
        with open(path, "w") as f: f.write("\n".join(new_lines))
        print("✅ Syntax fix applied.")
    else:
        print("ℹ️ Target block not found (maybe already fixed).")
else:
    print("❌ File not found: " + path)
'

# 3. Create run_mme_eval.sh
cat > run_mme_eval.sh <<EOF
#!/bin/bash
MME_ROOT="$MME_DIR"
RESULTS_DIR="$RESULTS_DIR"
BASE_DIR="$BASE_DIR"

# Set HF Home to avoid boot disk full
export HF_HOME="$HF_CACHE"
export PYTHONPATH="\$BASE_DIR:\$PYTHONPATH"

echo "🚀 Running MME Eval..."

# 1. Llava (Download from HF)
echo "🔍 [1/3] Running Llava (Baseline)"
python tools/mme_run_all.py --mme_root "\$MME_ROOT" --out_root "\$RESULTS_DIR/mme_llava" --model_path "liuhaotian/llava-v1.5-7b" --cfg config/default.yaml

# 2. VisionZip (Download from HF)
echo "🔍 [2/3] Running VisionZip"
python tools/mme_run_all.py --mme_root "\$MME_ROOT" --out_root "\$RESULTS_DIR/mme_visionzip" --model_path "liuhaotian/llava-v1.5-7b" --cfg config/default.yaml --dominant 54 --contextual 10

# 3. SparseZip (Download from HF)
echo "🔍 [3/3] Running SparseZip"
python tools/mme_run_all.py --mme_root "\$MME_ROOT" --out_root "\$RESULTS_DIR/mme_sparsezip" --model_path "liuhaotian/llava-v1.5-7b" --cfg config/sparsezip_mme.yaml

# Aggregate
python3 -c '
import os, csv
results_dir = "$RESULTS_DIR"
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
EOF
chmod +x run_mme_eval.sh

# 4. Run it
./run_mme_eval.sh
