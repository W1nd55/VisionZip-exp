#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting SparseZip VM Setup..."

# 1. Install Miniforge (Avoids Anaconda ToS issues)
if [ -d "$HOME/miniconda3" ]; then
    echo "⚠️  Found existing miniconda3. If you have ToS issues, run: rm -rf ~/miniconda3 and re-run this script."
fi

if ! command -v conda &> /dev/null; then
    echo "📦 Installing Miniforge..."
    mkdir -p ~/miniforge3
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O ~/miniforge3/miniforge.sh
    bash ~/miniforge3/miniforge.sh -b -u -p ~/miniforge3
    rm -rf ~/miniforge3/miniforge.sh
    source ~/miniforge3/bin/activate
    conda init bash
else
    echo "✅ Conda already installed."
    # Try to source typical paths
    source ~/miniforge3/bin/activate 2>/dev/null || source ~/miniconda3/bin/activate 2>/dev/null || true
fi

# Ensure conda is usable in this script
eval "$(conda shell.bash hook)"

# 2. Create/Activate Environment
echo "🐍 Creating conda environment 'sparsezip'..."
# Use -y to confirm, and handle case where env exists
conda create -n sparsezip python=3.10 -y || echo "Environment might already exist."

# Activate
conda activate sparsezip

# 3. Install Dependencies
echo "⬇️ Installing PyTorch and dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install transformers==4.37.2 tokenizers==0.15.2 accelerate==0.27.2
pip install sentencepiece protobuf huggingface_hub

# 4. Download Datasets
echo "📂 Downloading POPE and COCO datasets..."
python scripts/dataset_download.py --dataset pope coco_val --output_dir datasets

echo "🎉 Setup Complete! Run 'conda activate sparsezip' to start."
