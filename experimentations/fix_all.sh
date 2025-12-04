#!/bin/bash

# 1. Find MME Directory
echo "🔍 Searching for MME_Benchmark_release_version..."
MME_PATH=$(find ~/project/mittalshivam003 -type d -name "MME_Benchmark_release_version" -print -quit)

if [ -z "$MME_PATH" ]; then
    echo "❌ Could not find MME_Benchmark_release_version directory!"
    echo "Please edit run_mme_eval.sh manually to set MME_ROOT."
    MME_PATH="/home/w1nd519994824/project/mittalshivam003/MME_Benchmark_release_version"
else
    echo "✅ Found MME at: $MME_PATH"
fi

# 2. Create Default Config
mkdir -p config
cat > config/default.yaml <<EOF
dataset: mme
model_type: llava_vzip
temperature: 0.2
max_new_tokens: 128
EOF
echo "✅ Created config/default.yaml"

# 3. Create run_mme_eval.sh
cat > run_mme_eval.sh <<EOF
#!/bin/bash
MME_ROOT="$MME_PATH"
RESULTS_DIR="results"
MODEL_ROOT="/home/w1nd519994824/project/mittalshivam003/models"
BASE_DIR=\$(pwd)

mkdir -p "\$RESULTS_DIR"
source /home/w1nd519994824/miniforge3/etc/profile.d/conda.sh
conda activate env_sparsezip
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
echo "✅ Created run_mme_eval.sh"

# 4. Patch Python File
python3 -c '
import os
path = "models/SparseVLMs/llava/model/language_model/modelling_sparse_llama.py"
if os.path.exists(path):
    with open(path, "r") as f: content = f.read()
    target = "next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)"
    if target in content:
        # We replace the whole 3-line block by finding the unique middle line
        lines = content.splitlines()
        new_lines = []
        skip = 0
        for i, line in enumerate(lines):
            if skip > 0:
                skip -= 1
                continue
            if target in line:
                # This is the middle line. The previous line was "unfinished_sequences = ..."
                # We need to rewrite the previous line in new_lines
                prev = new_lines.pop() # remove "unfinished_sequences = unfinished_sequences.mul("
                indent = prev[:prev.find("unfinished")]
                new_lines.append(indent + "# Fix CUDA error")
                new_lines.append(indent + "is_eos = torch.isin(next_tokens, eos_token_id_tensor)")
                new_lines.append(indent + "unfinished_sequences = unfinished_sequences.mul((~is_eos).long())")
                skip = 1 # skip the closing parenthesis line
            else:
                new_lines.append(line)
        with open(path, "w") as f: f.write("\n".join(new_lines))
        print("✅ Patched modelling_sparse_llama.py")
    else:
        print("ℹ️ modelling_sparse_llama.py already patched or target not found")
else:
    print("❌ Could not find modelling_sparse_llama.py")
'

echo "🎉 Fixes applied! Now run: ./run_mme_eval.sh"
