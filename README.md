# VisionZip-exp
This is an exploratory research project focusing on optimizing the system-level performance and efficiency of the key VisionZip visual token selection architecture in the LLaVA-based multimodal LLM.

# 🧩 MME Benchmark Evaluation Guide

This document explains how to evaluate the [LLaVA + VisionZip] model on the MME Benchmark, measuring both accuracy and latency metrics.

---

## 📦 1. Dataset Preparation

1. Download the MME Benchmark dataset to your local directory or Hugging Face cache, for example:

```bash
mkdir -p ~/project/hf-cache/datasets--darkyarding--MME
# or download with huggingface-cli
huggingface-cli download darkyarding/MME --repo-type dataset --local-dir ~/project/hf-cache/datasets--darkyarding--MME
```

2. Make sure the dataset folder structure looks like this:

```
MME_Benchmark_release_version/MME_Benchmark/
 ├── artwork/
 ├── OCR/
 │    ├── questions_answers_YN/*.txt
 │    ├── images/...
 ├── color/
 ├── scene/
 ├── landmark/
 └── ...
```

Each subfolder corresponds to a subtask (OCR, scene, landmark, etc.).

---

## 🛠 2. Main Tool Scripts Overview

| Directory / File | Description |
|------------------|-------------|
| **`tools/mme_ann_from_txtpairs.py`** | Converts the `.txt` Q&A files in each MME subtask directory into a standard JSON annotation file (two Yes/No questions per image). |
| **`tools/mme_run_all.py`** | One-click evaluation script:<br>① Automatically calls `mme_ann_from_txtpairs.py` to build annotation JSONs for each subtask;<br>② Calls `evalkit.py` to run model inference;<br>③ Summarizes all subtask results into `mme_summary.csv`. |
| **`scripts/evalkit.py`** | The general evaluation entry. Supports `--dataset mme` for MME-format datasets. Internally uses `LlavaVisionZipModel` for inference. |
| **`scripts/dataset_mme.py`** | Defines how to load MME datasets (reads ann JSONs and expands Yes/No samples). |
| **`scripts/metric_mme.py`** | Defines MMEAcc and MMEAccPlus metrics:<br>single-question accuracy (ACC) and paired correctness (ACC+). |
| **`datasets/mme_eval_results/`** | Evaluation output directory (auto-created). Includes:<br>  • `ann/`: generated annotations<br>  • `results/outputs_*`: subtask results and summary files |
| **`visionzip/`** | The VisionZip module implementation (dominant/contextual token compression). |
| **`scripts/`** | General-purpose components (dataset, metric, abstract, evaluate, etc.). |

---

## 🚀 3. Example Usage

### (1) Evaluate a Single Subtask

```bash
python tools/mme_run_all.py   --mme_root ~/project/hf-cache/datasets--darkyarding--MME/snapshots/7056f44dac19de35e510b62d734bb2f7a6f64739/MME_Benchmark_release_version/MME_Benchmark   --out_root datasets/mme_eval_results   --only OCR   --evalkit scripts/evalkit.py
```

Process steps:
1. Automatically scans `.txt` Q&A files in the OCR directory;
2. Generates `datasets/mme_eval_results/ann/ann_mme_OCR.json`;
3. Calls `evalkit.py` to run model inference;
4. Writes results to `datasets/mme_eval_results/results/outputs_OCR/summary.csv`.

### (2) Evaluate All Subtasks

```bash
python tools/mme_run_all.py   --mme_root ~/project/hf-cache/datasets--darkyarding--MME/snapshots/7056f44dac19de35e510b62d734bb2f7a6f64739/MME_Benchmark_release_version/MME_Benchmark   --out_root datasets/mme_eval_results   --evalkit scripts/evalkit.py
```

After completion:
- Individual subtask results are stored in `results/outputs_<subtask>/summary.csv`.
- The overall summary file `mme_summary.csv` is saved under `datasets/mme_eval_results/`.
  It includes `mme_acc`, `mme_acc_plus`, latency (end2end_ms_avg, decode_tok_per_s, etc).

---

## ⚙️ 4. Evaluation Parameters

| Argument | Default | Description |
|-----------|----------|-------------|
| `--model_path` | `liuhaotian/llava-v1.5-7b` | Base model checkpoint |
| `--dominant` / `--contextual` | 54 / 10 | VisionZip compression settings |
| `--temperature` | 0.0 | Set to 0 for greedy decoding (stable Yes/No output) |
| `--max_new_tokens` | 16 | Limit output length for speed |
| `--warmup` | 5 | Warm-up iterations for stable CUDA timing |
| `--dataset` | `mme` | Dataset type (mme / vqa) |

---

## 📊 5. Output Format

Each subtask `summary.csv` example:

| metric | value |
|--------|-------|
| mme_acc | 0.885 |
| mme_acc_plus | 0.792 |
| end2end_ms_avg | 1320.5 |
| decode_tok_per_s | 42.1 |

Overall `mme_summary.csv` example:

| subtask | mme_acc | mme_acc_plus | end2end_ms_avg | decode_tok_per_s |
|----------|----------|---------------|----------------|------------------|
| OCR | 0.88 | 0.79 | 1320.5 | 42.1 |
| landmark | 0.91 | 0.85 | 1287.4 | 45.3 |
| ... | ... | ... | ... | ... |

---

## 🧭 6. Directory Overview

```
VisionZip-exp/
├── scripts/                 # Evaluation and general components
│   ├── evalkit.py           # Evaluation entrypoint
│   ├── dataset_mme.py       # MME dataset loader
│   ├── metric_mme.py        # MME metrics
│   ├── metric.py            # Common metrics (VQA etc.)
│   └── ...
├── tools/                   # Utility scripts
│   ├── mme_ann_from_txtpairs.py  # Convert .txt → .json
│   ├── mme_run_all.py            # Batch evaluation & summary
│   └── ...
├── datasets/
│   └── mme_eval_results/     # Auto-generated evaluation results
├── visionzip/                # VisionZip core implementation
└── README.md                 # This document
```

---

## ✅ 7. Notes

- Ensure `evalkit.py` supports `--dataset mme` and imports `MMEDataset`, `MMEAcc`, and `MMEAccPlus`.
- If you see `ModuleNotFoundError: No module named 'scripts'`, run from the project root or set `export PYTHONPATH=$PWD:$PYTHONPATH`.
- Use `--only OCR scene color` to evaluate selected subtasks.
- You can extend `mme_run_all.py` to add grouped summaries (e.g., Perceptual / Cognitive task groups).

