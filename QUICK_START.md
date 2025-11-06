# 🚀 Quick Start Guide - VisionZip Evaluation

## Step 1: Setup Environment (15 minutes)

**Choose ONE option:**

### Option A: Using venv (Built-in Python, Recommended)
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate (Windows)
venv\Scripts\activate

# 2. Activate (Linux/Mac)
source venv/bin/activate

# 3. Install LLaVA
cd models/LLaVA
pip install --upgrade pip
pip install -e .

# 4. Install VisionZip
cd ../VisionZip
pip install -e .

# 5. Go back to root
cd ../..
```

### Option B: Using conda (If you have it)
```bash
# 1. Create conda environment
conda create -n llava python=3.10 -y
conda activate llava

# 2. Install LLaVA
cd models/LLaVA
pip install --upgrade pip
pip install -e .

# 3. Install VisionZip
cd ../VisionZip
pip install -e .

# 4. Go back to root
cd ../..
```

### Option C: System Python (Not recommended, but works)
```bash
# Just install directly (risky if you have other projects)
cd models/LLaVA
pip install --upgrade pip
pip install -e .
cd ../VisionZip
pip install -e .
cd ../..
```

**Note:** You need Python 3.10 or 3.11. Check with: `python --version`

---

## Step 2: Test Setup (5 minutes)

```bash
# Test that everything works
python scripts/test_visionzip_single.py
```

**Expected output:**
- ✓ Model loaded successfully
- ✓ VisionZip applied successfully
- ✓ Setup verification complete!

---

## Step 3: Download POPE Dataset (10 minutes)

### Option A: Quick Download (Recommended)

1. **Download eval.zip** (contains question files):
   - Link: https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing
   - Extract to: `models/LLaVA/playground/data/eval/`

2. **Download POPE annotations**:
   ```bash
   cd models/LLaVA/playground/data/eval/pope
   # Download coco folder from: https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco
   # Place in: ./coco/
   ```

3. **Download COCO val2014 images**:
   - Link: http://images.cocodataset.org/zips/val2014.zip (~20GB)
   - Extract to: `models/LLaVA/playground/data/eval/pope/val2014/`

### Option B: Use Smaller Test Set (If 20GB is too large)

You can create a smaller test set with just 10-20 images for initial testing.

---

## Step 4: Run VisionZip Evaluation (30-60 minutes)

```bash
# From project root
python scripts/eval_pope_visionzip.py
```

**Or use the bash script:**
```bash
bash scripts/pope_visionzip.sh
```

**What it does:**
1. Loads LLaVA-7B model
2. Applies VisionZip (54 dominant + 10 contextual = 64 tokens)
3. Runs inference on POPE dataset
4. Calculates metrics (Accuracy, Precision, Recall, F1)

**Results saved to:**
`models/LLaVA/playground/data/eval/pope/answers/visionzip-v1.5-7b.jsonl`

---

## Step 5: Run SparseVLM Evaluation (Later)

Once VisionZip works, do the same for SparseVLM:

```bash
cd models/SparseVLMs
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh
```

**Note:** You may need to modify the script to use 7B model instead of 13B.

---

## Step 6: Compare Results

```bash
python scripts/compare_results.py
```

This will show a comparison table of VisionZip vs SparseVLM metrics.

---

## 🎯 Expected Timeline

- **Setup**: 30 minutes
- **First test**: 5 minutes
- **Dataset download**: 10-30 minutes (depending on internet)
- **VisionZip evaluation**: 30-60 minutes (depending on GPU)
- **SparseVLM evaluation**: 30-60 minutes
- **Comparison**: 5 minutes

**Total: ~2-3 hours** for complete reproduction

---

## ⚠️ Troubleshooting

### Out of Memory?
```bash
# Use 4-bit quantization (add to eval script)
--load-4bit
```

### Model download too slow?
- Models download automatically on first run
- 7B model is ~13GB
- Make sure you have stable internet

### Can't find dataset?
- Check paths in the script
- Make sure eval.zip is extracted correctly
- Verify image folder exists

### Need help?
- Check `REPRODUCTION_PLAN.md` for detailed steps
- Review error messages carefully
- Test with single image first

---

## 📊 What to Report

After running both evaluations, you should have:

1. **VisionZip results**: Accuracy, Precision, Recall, F1
2. **SparseVLM results**: Accuracy, Precision, Recall, F1
3. **Comparison table**: Side-by-side metrics
4. **Token counts**: VisionZip (64 tokens) vs SparseVLM (64 tokens)
5. **Inference time**: (optional, but useful)

---

## ✅ Success Checklist

- [ ] Environment setup complete
- [ ] Single image test passes
- [ ] POPE dataset downloaded
- [ ] VisionZip evaluation runs successfully
- [ ] SparseVLM evaluation runs successfully
- [ ] Comparison script shows results
- [ ] Results match paper (within reasonable margin)

---

**You're ready! Start with Step 1 above.** 🎉

