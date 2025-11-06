# 🐧 Linux GPU Server Guide

**Quick reference for running POPE evaluation on Linux GPU server.**

---

## ✅ STEP 1: Activate Environment (if not already active)

```bash
# If you already have venv created:
source venv/bin/activate

# If you need to create it (use python3 on Linux):
python3 -m venv venv
source venv/bin/activate
```

**Verify:** You should see `(venv)` at the start of your prompt.

---

## ✅ STEP 1.5: Install Dependencies

### Install PyTorch with CUDA (5-10 minutes)

```bash
# Make sure you're in the venv (should see (venv) in prompt)
# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.4)
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

**Verify PyTorch:**
```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Install LLaVA Dependencies (5-10 minutes)

```bash
# Install required packages
pip install transformers tokenizers sentencepiece shortuuid accelerate peft bitsandbytes pydantic markdown2 numpy scikit-learn gradio requests httpx uvicorn fastapi einops einops-exts timm pillow tqdm protobuf safetensors

# Install LLaVA (without dependencies since we already installed them)
cd models/LLaVA
pip install -e . --no-deps
cd ../..
```

### Install VisionZip (2 minutes)

```bash
cd models/VisionZip
pip install -e .
cd ../..
```

**Verify Installation:**
```bash
python3 -c "import torch; from visionzip import visionzip; print('✓ Setup complete!')"
```

---

## ✅ STEP 2: Verify GPU Access

```bash
# Check CUDA availability
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# OR use nvidia-smi
nvidia-smi
```

**Expected:** Should show your GPU model and CUDA version.

---

## ✅ STEP 3: Verify Dataset Structure

Your data should be at:
```
models/LLaVA/playground/eval/pope/
├── val2014/              ← Images (you're uploading this)
├── llava_pope_test.jsonl ← Question file (should exist)
├── coco/                 ← Annotations (should exist)
└── answers/              ← Results will go here
```

**Check if files exist:**
```bash
ls -la models/LLaVA/playground/eval/pope/
ls models/LLaVA/playground/eval/pope/val2014/ | head -5  # Check images
```

---

## ✅ STEP 4: Run VisionZip Evaluation

### Quick Test (10 samples - ~5 minutes):
```bash
python3 scripts/eval_pope_visionzip.py --max-samples 10
```

### Full Evaluation:
```bash
python3 scripts/eval_pope_visionzip.py
```

**What happens:**
- Loads LLaVA-7B model (downloads if first time)
- Applies VisionZip (dominant=54, contextual=10)
- Runs inference on POPE dataset
- Calculates POPE metrics (Accuracy, Precision, Recall, F1)

**Results saved to:** `models/LLaVA/playground/eval/pope/answers/visionzip-v1.5-7b.jsonl`

---

## ✅ STEP 5: Run SparseVLM Evaluation

### Quick Test (10 samples - ~5 minutes):
```bash
python3 scripts/eval_pope_sparsevlm.py --max-samples 10 --retained_tokens 64
```

### Full Evaluation:
```bash
python3 scripts/eval_pope_sparsevlm.py --retained_tokens 64
```

**What happens:**
- Loads LLaVA-7B model
- Applies SparseVLM sparsification (retained_tokens=64, matches VisionZip)
- Runs inference on POPE dataset
- Calculates POPE metrics

**Results saved to:** `models/LLaVA/playground/eval/pope/answers/sparsevlm-v1.5-7b.jsonl`

---

## ✅ STEP 6: Compare Results

```bash
python3 scripts/compare_results.py
```

**Output:** Comparison table showing VisionZip vs SparseVLM metrics.

---

## 🎯 Quick Command Reference

```bash
# Activate environment
source venv/bin/activate

# Quick test VisionZip (10 samples)
python3 scripts/eval_pope_visionzip.py --max-samples 10

# Full VisionZip evaluation
python3 scripts/eval_pope_visionzip.py

# Quick test SparseVLM (10 samples)
python3 scripts/eval_pope_sparsevlm.py --max-samples 10 --retained_tokens 64

# Full SparseVLM evaluation
python3 scripts/eval_pope_sparsevlm.py --retained_tokens 64

# Compare results
python3 scripts/compare_results.py
```

---

## ⚠️ Common Issues on Linux GPU Server

### "CUDA out of memory"
- The scripts auto-use 4-bit quantization if needed
- If still failing, reduce batch size or use smaller model

### "Permission denied"
- Make sure scripts are executable: `chmod +x scripts/*.py`

### "Module not found"
- Make sure environment is activated: `source venv/bin/activate`
- Install packages: See `STEP_BY_STEP.md` for installation steps

### "Dataset not found"
- Check path: `models/LLaVA/playground/eval/pope/` (NOT `playground/data/eval/pope/`)
- Verify files exist with `ls` commands above

---

## 📝 Notes

- **GPU required** - CPU inference is very slow
- **First run downloads models** - ~13GB for 7B model
- **Each evaluation takes 30-60 minutes** - be patient!
- **Use `--max-samples 10` for quick testing** before full run

---

## 🚀 Ready to Start?

1. Make sure `val2014/` folder is uploaded
2. Activate environment: `source venv/bin/activate`
3. Run quick test: `python3 scripts/eval_pope_visionzip.py --max-samples 10`
4. If successful, run full evaluation!

