#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting SparseZip VM Setup..."

# Get the current directory (assumed to be on the large disk)
WORK_DIR=$(pwd)
echo "📂 Working directory: $WORK_DIR"

# Set cache and temp directories to use the large disk
export TMPDIR="$WORK_DIR/tmp"
export PIP_CACHE_DIR="$WORK_DIR/cache/pip"
export CONDA_PKGS_DIRS="$WORK_DIR/cache/conda_pkgs"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

echo "💾 Configured cache/temp dirs on large disk:"
echo "   TMPDIR=$TMPDIR"
echo "   PIP_CACHE_DIR=$PIP_CACHE_DIR"

# 1. Install Miniforge to the local directory (Avoids filling up /home partition)
INSTALL_DIR="$WORK_DIR/miniforge3"

if [ -d "$HOME/miniconda3" ]; then
    echo "⚠️  Found existing miniconda3 in home. Ignoring it to use local installation."
fi

if [ ! -d "$INSTALL_DIR" ]; then
    echo "📦 Installing Miniforge to $INSTALL_DIR..."
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O "$WORK_DIR/miniforge.sh"
    bash "$WORK_DIR/miniforge.sh" -b -u -p "$INSTALL_DIR"
    rm "$WORK_DIR/miniforge.sh"
else
    echo "✅ Miniforge already installed in $INSTALL_DIR."
fi

# Source the new installation
source "$INSTALL_DIR/bin/activate"

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

# 2. Create/Activate Environment
echo "🐍 Creating conda environment 'sparsezip'..."
# Use -y to confirm, and handle case where env exists
conda create -n sparsezip python=3.10 -y || echo "Environment might already exist."

# Activate
conda activate sparsezip

# 3. Install Dependencies
echo "⬇️ Installing PyTorch and dependencies..."
# Ensure pip uses the large disk for caching
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --cache-dir "$PIP_CACHE_DIR"
pip install -r requirements.txt --cache-dir "$PIP_CACHE_DIR"
pip install transformers==4.37.2 tokenizers==0.15.2 accelerate==0.27.2 --cache-dir "$PIP_CACHE_DIR"
pip install sentencepiece protobuf huggingface_hub --cache-dir "$PIP_CACHE_DIR"

# 4. Download Datasets
echo "📂 Downloading POPE and COCO datasets..."
python scripts/dataset_download.py --dataset pope coco_val --output_dir datasets

# Create a helper script to activate the environment easily later
echo "#!/bin/bash" > activate_env.sh
echo "source $INSTALL_DIR/bin/activate" >> activate_env.sh
echo "conda activate sparsezip" >> activate_env.sh
chmod +x activate_env.sh

echo "🎉 Setup Complete!"
echo "👉 To start working, run: source ./activate_env.sh"
