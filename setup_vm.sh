#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting SparseZip VM Setup..."

# 1. Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "📦 Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda init bash
else
    echo "✅ Conda already installed."
fi

# 2. Create/Activate Environment
echo "🐍 Creating conda environment 'sparsezip'..."
source ~/miniconda3/bin/activate
conda create -n sparsezip python=3.10 -y || echo "Environment likely exists, skipping create."
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
