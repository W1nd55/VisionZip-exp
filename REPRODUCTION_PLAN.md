# VisionZip vs SparseVLM Reproduction Plan

## 🎯 Goal
Reproduce and compare VisionZip and SparseVLM results on the same dataset.

---

## 📋 Decision: Start with VisionZip

**Why VisionZip first?**
- ✅ No training needed (inference-only)
- ✅ Simpler code (just wraps LLaVA model)
- ✅ Works with smaller models (7B fits on laptop GPU)
- ✅ Faster to test

---

## 📊 Dataset Choice: POPE

**Why POPE?**
- ✅ Smallest dataset (~500 samples) - fast for testing
- ✅ Single-GPU friendly
- ✅ Self-contained evaluation (no external server)
- ✅ Used in both papers for comparison

**Dataset Info:**
- Size: ~500 questions
- Images: COCO val2014 (need to download)
- Evaluation: Accuracy, Precision, Recall, F1

---

## 🔧 Step-by-Step Plan

### **PHASE 1: Environment Setup** (30 min)

**Choose your environment manager:**

#### Option A: venv (Built-in, Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install LLaVA
cd models/LLaVA
pip install --upgrade pip
pip install -e .

# Install VisionZip
cd ../VisionZip
pip install -e .

# Install SparseVLM (for later)
cd ../SparseVLMs
pip install -e .
```

#### Option B: conda (If you have it)
```bash
# Create conda environment
conda create -n llava python=3.10 -y
conda activate llava

# Install LLaVA
cd models/LLaVA
pip install --upgrade pip
pip install -e .

# Install VisionZip
cd ../VisionZip
pip install -e .

# Install SparseVLM (for later)
cd ../SparseVLMs
pip install -e .
```

**Requirements:**
- Python 3.10 or 3.11
- pip
- GPU with CUDA (for inference)

4. **Verify GPU**
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

---

### **PHASE 2: Download Dataset** (15 min)

1. **Download eval.zip** (contains question files)
   - Link: https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing
   - Extract to: `models/LLaVA/playground/data/eval/`

2. **Download POPE data**
   ```bash
   cd models/LLaVA/playground/data/eval
   # Download coco folder from: https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco
   # Place in: ./pope/coco/
   ```

3. **Download COCO val2014 images**
   - Link: http://images.cocodataset.org/zips/val2014.zip
   - Extract to: `models/LLaVA/playground/data/eval/pope/val2014/`

---

### **PHASE 3: VisionZip Evaluation** (1-2 hours)

1. **Test on single image** (verify setup)
   ```bash
   python scripts/test_visionzip_single.py
   ```

2. **Run VisionZip on POPE**
   ```bash
   cd models/LLaVA
   bash scripts/v1_5/eval/pope_visionzip.sh
   ```

3. **Results saved to**: `playground/data/eval/pope/answers/visionzip-v1.5-7b.jsonl`

---

### **PHASE 4: SparseVLM Evaluation** (1-2 hours)

1. **Run SparseVLM on POPE**
   ```bash
   cd models/SparseVLMs
   bash scripts/v1_5/eval/pope.sh
   ```

2. **Results saved to**: `playground/data/eval/pope/answers/llava-v1.5-13b.jsonl` (or similar)

---

### **PHASE 5: Comparison** (30 min)

1. **Run comparison script**
   ```bash
   python scripts/compare_results.py
   ```

2. **Output**: Table comparing VisionZip vs SparseVLM metrics

---

## 📁 File Structure After Setup

```
cs769/
├── models/
│   ├── LLaVA/
│   │   └── playground/data/eval/pope/
│   │       ├── llava_pope_test.jsonl
│   │       ├── val2014/          (images)
│   │       ├── coco/             (annotations)
│   │       └── answers/
│   │           ├── visionzip-v1.5-7b.jsonl
│   │           └── sparsevlm-v1.5-7b.jsonl
│   ├── VisionZip/
│   └── SparseVLMs/
└── scripts/
    ├── test_visionzip_single.py
    ├── pope_visionzip.sh
    └── compare_results.py
```

---

## ⚙️ Model Configuration

**For Laptop GPU (Limited VRAM):**
- Model: `liuhaotian/llava-v1.5-7b` (7B parameters, ~14GB VRAM)
- VisionZip: `dominant=54, contextual=10` (64 tokens total)
- SparseVLM: `retained_tokens=64` (match VisionZip token count)

**If you have more VRAM:**
- Can use 13B model for better accuracy
- VisionZip: `dominant=191, contextual=30` (221 tokens)
- SparseVLM: `retained_tokens=192`

---

## 📈 Expected Results

Based on papers:
- **VisionZip**: ~95% performance with 10% tokens
- **SparseVLM**: Text-aware token selection
- **Comparison**: Should show similar accuracy, different approaches

---

## 🚨 Troubleshooting

**Out of Memory?**
- Use 4-bit quantization: Add `--load-4bit` flag
- Reduce batch size to 1
- Use smaller model (7B)

**Slow Inference?**
- Normal for first run (model downloads)
- Use `--temperature 0` for faster generation
- Reduce `max_new_tokens` if needed

---

## ✅ Next Steps After POPE

Once POPE works, you can add:
- ScienceQA (small, single-GPU)
- MME (small, single-GPU)
- TextVQA (medium, single-GPU)

