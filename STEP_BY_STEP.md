# 📋 Step-by-Step Execution Guide

Follow these steps **in order**. Copy and paste each command.

---

## ✅ STEP 1: Create Python Environment (2 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# If you see (venv) at the start of your prompt, you're good!
```

**Verify:** You should see `(venv)` at the start of your command prompt.

---

## ✅ STEP 2: Install PyTorch First (5 minutes)

```bash
# Install PyTorch with CUDA support (if you have GPU)
# Try CUDA 11.8 first (most common):
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# OR if above fails, try CUDA 12.1:
# pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# OR if you don't have GPU or CUDA:
# pip install torch==2.1.2 torchvision==0.16.2
```

**Verify:** `python -c "import torch; print('PyTorch:', torch.__version__)"`

**Note:** If you get "No matching distribution found", try the standard PyPI version:
```bash
pip install torch torchvision
```

---

## ✅ STEP 3: Install LLaVA (5-10 minutes)

```bash
cd models/LLaVA
pip install --upgrade pip

# If you get "torch==2.1.2 not found" error, install dependencies manually:
# (Use latest compatible versions to avoid compilation issues)
# Option 1: Install without strict version pins (easiest):
pip install transformers tokenizers sentencepiece shortuuid accelerate peft bitsandbytes pydantic markdown2 numpy scikit-learn gradio requests httpx uvicorn fastapi einops einops-exts timm pillow tqdm protobuf safetensors

# Option 2: If you need specific versions, force binary wheels only:
# pip install --only-binary=:all: transformers==4.37.2 tokenizers sentencepiece shortuuid accelerate peft pydantic markdown2 numpy scikit-learn gradio==4.16.0 requests httpx uvicorn fastapi einops einops-exts timm pillow tqdm protobuf

# Then install LLaVA without deps (since you already have torch)
pip install -e . --no-deps

# OR if you want to try normal installation first:
# pip install -e .
cd ../..
```

**Wait for:** Installation to complete (may take a few minutes).

**Note:** If you have a newer PyTorch version (like 2.9.0), that's fine - it should work with LLaVA.

---

## ✅ STEP 4: Install VisionZip (2 minutes)

```bash
cd models/VisionZip
pip install -e .
cd ../..
```

---

## ✅ STEP 5: Test Your Setup (5 minutes)

```bash
python scripts/test_visionzip_single.py
```

**Expected output:**
- ✓ Model loaded successfully
- ✓ VisionZip applied successfully
- ✓ Setup verification complete!

**If errors:** Check that you have internet (model downloads on first run).

---

## ✅ STEP 6: Download Dataset (10-30 minutes)

### 5a. Download eval.zip

1. Go to: https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing
2. Download `eval.zip`
3. Extract to: `models/LLaVA/playground/data/eval/`

**Verify:** Check that `models/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl` exists.

### 5b. Download POPE annotations

1. Go to: https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco
2. Download the `coco` folder
3. Place in: `models/LLaVA/playground/data/eval/pope/coco/`

**Verify:** Check that `models/LLaVA/playground/data/eval/pope/coco/` exists.

### 5c. Download COCO images (LARGE - 20GB)

1. Download: http://images.cocodataset.org/zips/val2014.zip
2. Extract to: `models/LLaVA/playground/data/eval/pope/val2014/`

**Note:** This is ~20GB. If too large, you can test with a smaller subset first.

**Verify:** Check that `models/LLaVA/playground/data/eval/pope/val2014/` contains image files.

---

## ✅ STEP 7: Run VisionZip Evaluation (30-60 minutes)

### Quick Test (10 samples - ~5 minutes):
```bash
python scripts/eval_pope_visionzip.py --max-samples 10
```

### Full Evaluation:
```bash
python scripts/eval_pope_visionzip.py
```

**What it does:**
- Loads LLaVA-7B model (downloads if first time)
- Applies VisionZip
- Runs inference on POPE dataset (or random samples if --max-samples set)
- Shows progress bar (tqdm)
- Calculates metrics

**Results saved to:** `models/LLaVA/playground/data/eval/pope/answers/visionzip-v1.5-7b.jsonl`

**Wait for:** Completion message "✓ Complete!"

**Note:** Random sampling with `--max-samples` works correctly - it uses `random.sample()` which ensures no duplicates and proper randomness.

---

## ✅ STEP 8: Install SparseVLM (2 minutes)

**First, ensure correct transformers version (in venv):**
```bash
pip install tokenizers  # Install newer tokenizers with pre-built wheels first
pip install transformers==4.37.2 --no-deps  # Install transformers without dependencies
```

**Then install SparseVLM:**
```bash
cd models/SparseVLMs
pip install -e . --no-deps
cd ../..
```

**Note:** SparseVLM requires `transformers==4.37.2`. If you have a newer version, it will cause import errors. Use `--no-deps` to avoid reinstalling PyTorch.

---

## ✅ STEP 9: Run SparseVLM Evaluation (30-60 minutes)

### Quick Test (10 samples - ~5 minutes):
```bash
python scripts/eval_pope_sparsevlm.py --max-samples 10 --retained_tokens 64
```

### Full Evaluation:
```bash
python scripts/eval_pope_sparsevlm.py --retained_tokens 64
```

**What it does:**
- Loads LLaVA-7B model (downloads if first time)
- Applies SparseVLM sparsification (retained_tokens=64)
- Runs inference on POPE dataset
- Shows progress bar (tqdm)
- Calculates POPE metrics

**Results saved to:** `models/LLaVA/playground/eval/pope/answers/sparsevlm-v1.5-7b.jsonl`

**Note:** `--retained_tokens 64` matches VisionZip's total tokens (54+10=64). Options: 64, 128, or 192.

---

## ✅ STEP 10: Compare Results (2 minutes)

```bash
python scripts/compare_results.py
```

**Output:** Comparison table showing VisionZip vs SparseVLM metrics.

---

## 🎯 Quick Reference Checklist

- [ ] Step 1: Environment created
- [ ] Step 2: LLaVA installed
- [ ] Step 3: VisionZip installed
- [ ] Step 4: Test passed
- [ ] Step 5: Dataset downloaded
- [ ] Step 6: VisionZip evaluation complete
- [ ] Step 7: SparseVLM installed
- [ ] Step 8: SparseVLM evaluation complete
- [ ] Step 9: Comparison done

---

## ⚠️ Common Issues

### "Out of Memory"
```bash
# Use 4-bit quantization (modify eval script to add --load-4bit)
```

### "Model download too slow"
- Normal on first run
- 7B model is ~13GB
- Make sure you have stable internet

### "Can't find dataset"
- Double-check paths in Step 5
- Make sure you extracted files correctly

### "Command not found"
- Make sure you activated the environment (Step 1)
- You should see `(venv)` in your prompt

---

## 📝 Notes

- **First run is slow** - models download automatically
- **GPU required** - CPU inference is very slow
- **Dataset is large** - COCO images are ~20GB
- **Each evaluation takes 30-60 minutes** - be patient!

---

**Ready? Start with Step 1!** 🚀

