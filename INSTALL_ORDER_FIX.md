# 🔧 Fix: Tokenizers Dependency Conflict

## Problem
- `transformers==4.37.2` requires `tokenizers==0.15.2` (needs Rust)
- But we want `tokenizers` latest (has wheels)

## Solution: Install in Correct Order

Install tokenizers first with a compatible version, then transformers:

```bash
# Step 1: Install tokenizers first (with a version that has wheels and is compatible)
pip install "tokenizers>=0.15.0,<0.16.0" --upgrade

# If that still fails, try installing a specific version with wheels:
# pip install tokenizers==0.15.2 --only-binary=:all:

# OR use a newer transformers version that works with newer tokenizers:
pip install transformers tokenizers sentencepiece shortuuid accelerate peft pydantic markdown2 numpy scikit-learn gradio requests httpx uvicorn fastapi einops einops-exts timm pillow tqdm
```

---

## Better Solution: Use Compatible Versions

Since you're on Python 3.13 (newer), use compatible versions:

```bash
# Install all packages without strict version pins (use compatible versions)
pip install transformers tokenizers sentencepiece shortuuid accelerate peft pydantic markdown2 numpy scikit-learn gradio requests httpx uvicorn fastapi einops einops-exts timm pillow tqdm
```

Then when installing LLaVA, it should work with these versions.

---

## Alternative: Force Binary Installation

```bash
# Force pip to only use pre-built wheels (no compilation)
pip install --only-binary=:all: transformers==4.37.2 tokenizers sentencepiece shortuuid accelerate peft pydantic markdown2 numpy scikit-learn gradio==4.16.0 requests httpx uvicorn fastapi einops einops-exts timm pillow tqdm
```

---

## If Still Failing: Skip Exact Versions

```bash
# Just install latest compatible versions
pip install transformers tokenizers sentencepiece shortuuid accelerate peft pydantic markdown2 numpy scikit-learn gradio requests httpx uvicorn fastapi einops einops-exts timm pillow tqdm
```

Then install LLaVA with `--no-deps` flag.

