# 🚀 Commands to Run (Copy & Paste)

**Run these commands in order from your project root.**

---

## 1️⃣ Setup Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 2️⃣ Install Packages

```bash
cd models/LLaVA
pip install --upgrade pip
pip install -e .
cd ../VisionZip
pip install -e .
cd ../..
```

---

## 3️⃣ Test Setup

```bash
python scripts/test_visionzip_single.py
```

**✓ If you see "Setup verification complete!" → Continue**  
**✗ If errors → Fix them first**

---

## 4️⃣ Download Dataset

**Do this manually:**
1. Download `eval.zip` from: https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing
2. Extract to: `models/LLaVA/playground/data/eval/`
3. Download POPE `coco` folder from: https://github.com/AoiDragon/POPE
4. Place in: `models/LLaVA/playground/data/eval/pope/coco/`
5. Download COCO val2014: http://images.cocodataset.org/zips/val2014.zip
6. Extract to: `models/LLaVA/playground/data/eval/pope/val2014/`

---

## 5️⃣ Run VisionZip Evaluation

```bash
python scripts/eval_pope_visionzip.py
```

**Wait:** 30-60 minutes (first time longer due to model download)

---

## 6️⃣ Install SparseVLM

```bash
cd models/SparseVLMs
pip install -e .
cd ../..
```

---

## 7️⃣ Run SparseVLM Evaluation

```bash
cd models/SparseVLMs
python -m llava.eval.model_vqa_loader --model-path liuhaotian/llava-v1.5-7b --question-file ../LLaVA/playground/data/eval/pope/llava_pope_test.jsonl --image-folder ../LLaVA/playground/data/eval/pope/val2014 --answers-file ../LLaVA/playground/data/eval/pope/answers/sparsevlm-v1.5-7b.jsonl --temperature 0 --conv-mode vicuna_v1 --retained_tokens 64
cd ../..
```

**Wait:** 30-60 minutes

---

## 8️⃣ Compare Results

```bash
python scripts/compare_results.py
```

---

## ✅ Done!

You now have:
- VisionZip results
- SparseVLM results  
- Comparison table

---

## 🆘 If Something Fails

**Check:**
- Environment activated? (see `(venv)` in prompt)
- Dataset downloaded? (check file paths)
- GPU available? (needed for inference)
- Internet connection? (for model download)

**See:** `STEP_BY_STEP.md` for detailed troubleshooting

