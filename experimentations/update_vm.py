import os
import sys
import subprocess

# Base directory (current dir)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def write_file(rel_path, content):
    path = os.path.join(BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"✅ Updated {rel_path}")

def find_mme_root():
    # Try common locations
    candidates = [
        "/home/w1nd519994824/project/mittalshivam003/MME_Benchmark_release_version",
        "/home/w1nd519994824/project/mittalshivam003/MME",
        "/home/w1nd519994824/project/mittalshivam003/data/MME_Benchmark_release_version",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    
    # Try to find it via find command (max depth 3 to avoid taking too long)
    print("⚠️ MME root not found in common locations. Searching...")
    try:
        cmd = ["find", "/home/w1nd519994824/project/mittalshivam003", "-maxdepth", "3", "-type", "d", "-name", "MME_Benchmark_release_version"]
        result = subprocess.check_output(cmd).decode().strip().split('\n')[0]
        if result:
            return result
    except Exception:
        pass
    
    return "/PATH/TO/MME_Benchmark_release_version_NOT_FOUND"

def patch_llama_file():
    path = os.path.join(BASE_DIR, "models/SparseVLMs/llava/model/language_model/modelling_sparse_llama.py")
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return

    with open(path, "r") as f:
        content = f.read()

    target_block = """                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )"""
    
    replacement_block = """                # Simplified EOS check
                is_eos = torch.isin(next_tokens, eos_token_id_tensor)
                unfinished_sequences = unfinished_sequences.mul((~is_eos).long())"""

    if target_block in content:
        new_content = content.replace(target_block, replacement_block)
        with open(path, "w") as f:
            f.write(new_content)
        print("✅ Patched modelling_sparse_llama.py (CUDA fix applied)")
    elif "is_eos = torch.isin" in content:
        print("ℹ️ modelling_sparse_llama.py is already patched.")
    else:
        print("⚠️ Could not find target block in modelling_sparse_llama.py. It might be different than expected.")

# --- File Contents ---

# 1. Default Config (needed for mme_run_all.py)
DEFAULT_YAML = """
dataset: mme
model_type: llava_vzip
temperature: 0.2
max_new_tokens: 128
"""

# 2. Run Script
RUN_MME_EVAL_SH_TEMPLATE = r"""#!/bin/bash

# Define paths
MME_ROOT="{MME_ROOT}"
RESULTS_DIR="{BASE_DIR}/results"
MODEL_ROOT="/home/w1nd519994824/project/mittalshivam003/models"

# Ensure directories exist
mkdir -p "$RESULTS_DIR"

# Activate environment
source /home/w1nd519994824/miniforge3/etc/profile.d/conda.sh
conda activate env_sparsezip

# Fix PYTHONPATH so scripts.metric can be imported
export PYTHONPATH="{BASE_DIR}:$PYTHONPATH"

echo "============================================"
echo "🚀 Starting MME Evaluations"
echo "============================================"
echo "MME_ROOT: $MME_ROOT"

if [ ! -d "$MME_ROOT" ]; then
    echo "❌ Error: MME_ROOT directory does not exist!"
    exit 1
fi

# 1. Llava (Baseline)
echo "🔍 [1/4] Running MME - Llava (Baseline)"
python tools/mme_run_all.py \
    --mme_root "$MME_ROOT" \
    --out_root "$RESULTS_DIR/mme_llava" \
    --model_path "$MODEL_ROOT/llava-v1.5-7b" \
    --cfg config/default.yaml

# 2. VisionZip
echo "🔍 [2/4] Running MME - VisionZip"
python tools/mme_run_all.py \
    --mme_root "$MME_ROOT" \
    --out_root "$RESULTS_DIR/mme_visionzip" \
    --model_path "$MODEL_ROOT/llava-v1.5-7b" \
    --cfg config/default.yaml \
    --dominant 54 --contextual 10

# 3. SparseVLM
echo "🔍 [3/4] Running MME - SparseVLM"
python tools/mme_run_all.py \
    --mme_root "$MME_ROOT" \
    --out_root "$RESULTS_DIR/mme_sparsevlm" \
    --model_path "$MODEL_ROOT/SparseVLM_7B_224" \
    --model_type sparsevlm \
    --cfg config/default.yaml

# 4. SparseZip (Optimized)
echo "🔍 [4/4] Running MME - SparseZip (Optimized)"
python tools/mme_run_all.py \
    --mme_root "$MME_ROOT" \
    --out_root "$RESULTS_DIR/mme_sparsezip" \
    --cfg config/sparsezip_mme.yaml

echo ""
echo "✅ All MME evaluations complete!"
echo ""
echo "📈 Aggregating MME results into summary_final.csv..."

# Create summary CSV aggregator
python3 - << 'EOF_PYTHON'
import os
import csv
import glob

results_dir = "{BASE_DIR}/results"
output_csv = os.path.join(results_dir, "summary_final.csv")

# Define the models we ran
models = [
    {"name": "Llava (Baseline)", "dir": "mme_llava"},
    {"name": "VisionZip", "dir": "mme_visionzip"},
    {"name": "SparseVLM", "dir": "mme_sparsevlm"},
    {"name": "SparseZip", "dir": "mme_sparsezip"},
]

final_results = []

for model in models:
    # Look for mme_summary.csv in the model's output directory
    summary_file = os.path.join(results_dir, model["dir"], "mme_summary.csv")
    
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                reader = csv.DictReader(f)
                # Assuming mme_summary.csv has one row of results or we take the last one
                for row in reader:
                    # Add model name to the row
                    row["Model"] = model["name"]
                    final_results.append(row)
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")
    else:
        print(f"Warning: Results not found for {model['name']} at {summary_file}")

# Write combined results
if final_results:
    # Get all fieldnames from all results (some might be missing keys)
    fieldnames = ["Model"]
    all_keys = set()
    for res in final_results:
        all_keys.update(res.keys())
    
    # Sort keys for consistent output, keeping Model first
    sorted_keys = sorted([k for k in all_keys if k != "Model"])
    fieldnames.extend(sorted_keys)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_results)
    
    print(f"Successfully wrote aggregated results to {output_csv}")
else:
    print("No results found to aggregate.")

EOF_PYTHON

echo ""
echo "🎉 Evaluation complete! Check $RESULTS_DIR/summary_final.csv for results."
"""

# --- Execution ---

if __name__ == "__main__":
    print("🚀 Starting VM Update (Local Mode)...")
    
    # 0. Find MME Root
    mme_root = find_mme_root()
    print(f"📂 Found MME Root: {mme_root}")

    # 1. Create Default Config
    write_file("config/default.yaml", DEFAULT_YAML)

    # 2. Update run_mme_eval.sh
    run_script = RUN_MME_EVAL_SH_TEMPLATE.format(MME_ROOT=mme_root, BASE_DIR=BASE_DIR)
    write_file("run_mme_eval.sh", run_script)
    os.chmod(os.path.join(BASE_DIR, "run_mme_eval.sh"), 0o755)

    # 3. Patch modelling_sparse_llama.py
    patch_llama_file()
    
    print("\n✅ Update finished! You can now run ./run_mme_eval.sh")
