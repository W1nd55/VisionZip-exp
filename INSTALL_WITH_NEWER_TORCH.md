# 🔧 Fix: PyTorch Version Mismatch

## Problem
- You have PyTorch 2.9.0 installed
- LLaVA requires torch==2.1.2 (older)
- Pip can't find torch==2.1.2

## Solution: Install Dependencies Manually

Since you already have PyTorch 2.9.0 (which should work fine), let's install the other dependencies manually:

```bash
# Make sure you're in venv and in the project root
# (venv) PS > cd .. (to get back to root if you're in models/LLaVA)

# Install all required packages
pip install transformers==4.37.2
pip install tokenizers==0.15.1 sentencepiece==0.1.99 shortuuid
pip install accelerate==0.21.0 peft bitsandbytes
pip install pydantic markdown2 numpy scikit-learn==1.2.2
pip install gradio==4.16.0 gradio_client==0.8.1
pip install requests httpx==0.24.0 uvicorn fastapi
pip install einops==0.6.1 einops-exts==0.0.4 timm==0.6.13
pip install pillow tqdm
```

---

## Alternative: Install LLaVA Without Version Check

```bash
# Go to LLaVA directory
cd models/LLaVA

# Install without checking torch version
pip install -e . --no-deps
pip install transformers==4.37.2 tokenizers==0.15.1 sentencepiece==0.1.99 shortuuid accelerate==0.21.0 peft pydantic markdown2 numpy scikit-learn==1.2.2 gradio==4.16.0 requests httpx==0.24.0 uvicorn fastapi einops==0.6.1 einops-exts==0.0.4 timm==0.6.13 pillow tqdm

cd ../..
```

---

## Verify

```bash
python -c "import torch; import transformers; print('✓ PyTorch:', torch.__version__); print('✓ Transformers:', transformers.__version__)"
```

Then test:
```bash
python scripts/test_visionzip_single.py
```

---

## Note

PyTorch 2.9.0 should work fine with LLaVA - the code is generally compatible. The version requirement is just for reproducibility.

