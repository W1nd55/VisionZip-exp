#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting SparseZip VM Setup (Local Disk Mode)..."

# Ensure we are on the large disk
WORK_DIR=$(pwd)
echo "📂 Working directory: $WORK_DIR"

# CLEANUP: Remove broken/old installations from previous attempts
if [ -d "$WORK_DIR/miniforge3" ]; then
    echo "🗑️  Removing broken 'miniforge3' directory from previous install..."
    rm -rf "$WORK_DIR/miniforge3"
fi

# Respect externally provided cache/temp env vars (so users can point to a large disk)
export TMPDIR="${TMPDIR:-$WORK_DIR/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_DIR/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORK_DIR/cache/conda_pkgs}"
export HF_HOME="${HF_HOME:-$WORK_DIR/cache/huggingface}"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME"

echo "💾 Configured directories (effective):"
echo "   TMPDIR=$TMPDIR"
echo "   PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "   CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"
echo "   HF_HOME=$HF_HOME"

# 1. Install Miniforge LOCALLY (Ignore system/home conda)
INSTALL_DIR="$WORK_DIR/miniforge_local"

# Deactivate any existing conda (if installed)
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda deactivate 2>/dev/null || true
    conda deactivate 2>/dev/null || true
fi

if [ ! -d "$INSTALL_DIR" ]; then
    echo "📦 Installing Miniforge to $INSTALL_DIR..."
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O "$WORK_DIR/miniforge.sh"
    bash "$WORK_DIR/miniforge.sh" -b -u -p "$INSTALL_DIR"
    rm "$WORK_DIR/miniforge.sh"
else
    echo "✅ Miniforge already installed in $INSTALL_DIR."
fi

# Source the NEW local installation
source "$INSTALL_DIR/bin/activate"
eval "$(conda shell.bash hook)"

echo "🐍 Using Conda: $(which conda)"

# 2. Create/Activate Environment (Prefix install to keep it local)
ENV_DIR="$WORK_DIR/env_sparsezip"
echo "🐍 Creating conda environment in $ENV_DIR..."

if [ ! -d "$ENV_DIR" ]; then
    conda create -p "$ENV_DIR" python=3.10 -y
else
    echo "Environment directory exists."
fi

conda activate "$ENV_DIR"

# 3. Install Dependencies
echo "⬇️ Installing PyTorch and dependencies..."
# Ensure standard temp envs point to our large-disk tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$WORK_DIR/cache"

echo "Using TMPDIR=$TMPDIR, PIP_CACHE_DIR=$PIP_CACHE_DIR, CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"

# Install PyTorch and related wheels (allow caching to avoid repeated downloads);
# avoid --no-cache-dir because we want pip to reuse downloaded wheels in $PIP_CACHE_DIR
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --cache-dir "$PIP_CACHE_DIR"

# Install repository-specific requirements (prefer visionzip-specific file if present)
if [ -f "$WORK_DIR/requirements_A40_visionzip.txt" ]; then
    pip install -r "$WORK_DIR/requirements_A40_visionzip.txt" --cache-dir "$PIP_CACHE_DIR"
elif [ -f "$WORK_DIR/requirements_A40.txt" ]; then
    pip install -r "$WORK_DIR/requirements_A40.txt" --cache-dir "$PIP_CACHE_DIR"
else
    echo "[warn] No requirements_A40*.txt found; skipping pip install -r"
fi

# Core transformers and utility packages
pip install transformers==4.37.2 tokenizers==0.15.2 accelerate==0.27.2 --cache-dir "$PIP_CACHE_DIR"
pip install sentencepiece protobuf huggingface_hub --cache-dir "$PIP_CACHE_DIR"

# 4. Download Datasets
echo "📂 Downloading POPE and COCO datasets..."
python scripts/dataset_download.py --dataset pope coco_val --output_dir datasets

# Create helper script
echo "#!/bin/bash" > activate_env.sh
echo "source $INSTALL_DIR/bin/activate" >> activate_env.sh
echo "conda activate $ENV_DIR" >> activate_env.sh
echo "export TMPDIR=$TMPDIR" >> activate_env.sh
echo "export PIP_CACHE_DIR=$PIP_CACHE_DIR" >> activate_env.sh
echo "export HF_HOME=$HF_HOME" >> activate_env.sh
chmod +x activate_env.sh

echo "🎉 Setup Complete!"
echo "👉 To start working, run: source ./activate_env.sh"
