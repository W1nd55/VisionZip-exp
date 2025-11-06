# 🔧 Install Missing Dependencies

## Quick Fix - Install All Required Packages

Run these commands **one by one**:

```bash
# 1. Install transformers (required)
pip install transformers==4.37.2

# 2. Install other essential packages
pip install tokenizers==0.15.1 sentencepiece==0.1.99 shortuuid
pip install accelerate==0.21.0
pip install pillow tqdm

# 3. Install remaining dependencies
pip install pydantic markdown2 numpy scikit-learn==1.2.2
pip install einops==0.6.1 einops-exts==0.0.4 timm==0.6.13
```

---

## Better Option: Reinstall LLaVA Properly

Since LLaVA should install all these automatically, let's do it properly:

```bash
# Make sure you're in venv
# (you should see (venv) in your prompt)

# Go to LLaVA directory
cd models/LLaVA

# Reinstall with all dependencies
pip install --upgrade pip
pip install -e .

# Go back
cd ../..
```

This should install all required packages including transformers.

---

## Verify Installation

```bash
python -c "import torch; import transformers; print('✓ PyTorch:', torch.__version__); print('✓ Transformers:', transformers.__version__)"
```

Then try your test again:
```bash
python scripts/test_visionzip_single.py
```

