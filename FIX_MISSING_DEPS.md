# 🔧 Fix: Missing Dependencies

## Quick Fix - Install PyTorch and Dependencies

Run these commands **in order**:

```bash
# Make sure you're in your venv
venv\Scripts\activate

# Install PyTorch (with CUDA support if you have GPU)
# For CUDA 11.8 or 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OR for CPU only (slower):
# pip install torch torchvision torchaudio

# Install other dependencies
pip install transformers==4.37.2
pip install accelerate==0.21.0
pip install pillow
pip install tqdm
```

---

## Complete Installation (Recommended)

Since LLaVA installation should have installed these, let's reinstall properly:

```bash
# Make sure you're in venv
venv\Scripts\activate

# 1. Install PyTorch first
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 2. Reinstall LLaVA (this will install all dependencies)
cd models/LLaVA
pip install -e .
cd ../..

# 3. Install VisionZip
cd models/VisionZip
pip install -e .
cd ../..
```

---

## Check Your Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**Expected output:**
```
PyTorch: 2.1.2
CUDA available: True  (or False if no GPU)
```

---

## If Still Having Issues

Try installing everything explicitly:

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.2 tokenizers==0.15.1 sentencepiece==0.1.99 shortuuid
pip install accelerate==0.21.0 peft bitsandbytes
pip install pydantic markdown2 numpy scikit-learn==1.2.2
pip install gradio==4.16.0 requests httpx==0.24.0 uvicorn fastapi
pip install einops==0.6.1 einops-exts==0.0.4 timm==0.6.13
pip install pillow tqdm
```

Then try your test again:
```bash
python scripts/test_visionzip_single.py
```

