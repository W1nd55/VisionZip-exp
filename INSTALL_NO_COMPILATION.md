# 🔧 Fix: Avoid Compilation Issues

## Problem
- `sentencepiece==0.1.99` needs C++ build tools on Windows
- `tokenizers==0.15.1` needs Rust

## Solution: Use Compatible Versions with Pre-built Wheels

Install packages that have pre-built wheels (no compilation needed):

```bash
# Install core packages first (these have pre-built wheels)
pip install transformers==4.37.2
pip install tokenizers  # Use latest (has wheels)
pip install sentencepiece  # Use latest (has wheels, no compilation)
pip install shortuuid accelerate peft pydantic markdown2 numpy scikit-learn
pip install gradio requests httpx uvicorn fastapi
pip install einops einops-exts timm pillow tqdm
```

**OR install all at once:**

```bash
pip install transformers==4.37.2 tokenizers sentencepiece shortuuid accelerate peft pydantic markdown2 numpy scikit-learn gradio==4.16.0 requests httpx uvicorn fastapi einops einops-exts timm pillow tqdm
```

---

## Key Changes

- `tokenizers` → Use latest (no version pin)
- `sentencepiece` → Use latest (no version pin)
- Other packages → Use compatible versions or latest

---

## Then Install LLaVA

```bash
cd models/LLaVA
pip install -e . --no-deps
cd ../..
```

---

## Verify

```bash
python -c "import transformers; import sentencepiece; print('✓ Transformers:', transformers.__version__); print('✓ SentencePiece:', sentencepiece.__version__)"
```

---

## Why This Works

- Newer versions have pre-built wheels for Windows
- No compilation needed (no Rust, no C++)
- Compatible with LLaVA code
- Much faster installation

