# VisionZip/SparseVLM Modular Evaluation Framework

> This repository provides a modular and customizable framework for evaluating LLMs, specifically designed to handle models like **VisionZip** and **SparseVLM** on datasets such as **MME** and **POPE**. You can place custom model utility modifications (e.g., monkey patches) within the `utils/` directory.

-----

## MME Evaluation README

> Updated: 2025-11-08 06:35

This README explains how to:

1)  Sync Git submodules and prepare the results directory.
2)  Download the **MME** dataset using `dataset_download.py`.
3)  Customize evaluation modules in `dataset / metric / model`.
4)  Use `tools/mme_run_all.py` for one-click evaluation, including the `--only` option for selecting subtasks.

-----

## 0\. Quick Preparation

```bash
# Sync all git submodules
git submodule update --init --recursive

# Create the root directory for evaluation results (recommended inside/outside the repo)
mkdir -p eval_results
```

> **Recommended Environment**: Python 3.10+, `torch`, `transformers`, `Pillow`, `huggingface_hub`, etc.
> If accessing restricted/gated HF resources, please log in first:
>
> ```bash
> huggingface-cli login
> ```

-----

## 1\) Download MME Dataset (using `dataset_download.py`)

Script location: `scripts/dataset_download.py`
It includes `DATASET_CONFIGS["mme"]` by default, which fetches data from **Hugging Face: `darkyarding/MME`** and **automatically extracts** `MME_Benchmark_release_version.zip`.

### Basic Usage

```bash
python scripts/dataset_download.py   --dataset mme   --output_dir ./datasets
```

Upon completion, the typical directory structure is:

```
datasets/
  mme/
    MME_Benchmark_release_version/
      MME_Benchmark/
        OCR/
        ...  (contains several subtask directories)
```

> Common optional arguments:
>
>   - `--list`: Only lists available dataset names, does not download.
>   - `--no-extract`: Downloads only the zip file, does not extract.
>   - `--keep-zip`: Retains the zip archive after extraction.
>   - `--skip-auth-check`: Skips the login check (manual login is still required if the dataset is gated).

-----

## 2\) Customizing Evaluation in `dataset / metric / model`

This repository modularizes the evaluation process:

  - **`dataset.py`**: Defines data reading and sample structure (must return a list of `Sample` objects). `MMEDataset` is provided, supporting *paired annotations* and *flat annotations* schema, and automatically appends the suffix "`Please answer yes or no.`" to MME questions.
  - **`metric.py`**: Defines metrics. Built-in metrics include:
      - `MMEAcc`: Per-question accuracy (ACC).
      - `MMEAccPlus`: Per-image paired ACC+ (counted as 1 only if both positive/negative questions for the same `image_id` are answered correctly).
      - Also includes common VQA/POPE metrics (for example).
  - **`model.py`**: Encapsulates the single-sample inference pipeline (e.g., `LlavaVisionZipModel`, covering VisionZip patching and loading strategies), and uniformly returns the **predicted text** and **stage timings** (used to calculate e2e, prefill, decode, and other latency metrics).

### Customizing Dataset (Example)

```python
# dataset.py
from scripts.abstract import BaseDataset, Sample

class MyMMEDataset(BaseDataset):
    def __init__(self, ann_path: str, limit: int | None = None):
        # Read JSON -> Construct list of Sample(...) objects
        ...

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]
```

### Customizing Metric (Example)

```python
# metric.py
from scripts.abstract import BaseMetric, Sample

class MyBinaryAcc(BaseMetric):
    def __init__(self):
        self.n = 0; self.c = 0
    def update(self, sample: Sample, pred_text: str, timings_ms):
        # Normalize pred_text to yes/no and compare with sample.answers[0]
        ...
    def compute(self):
        return {"my_binary_acc": self.c / self.n if self.n else 0.0}
```

### Customizing Model (Example)

```python
# model.py
from scripts.abstract import BaseModel, Sample

class MyModel(BaseModel):
    def __init__(self, ...):
        ...
    def run(self, sample: Sample):
        # Return (pred_text, logs_dict); logs_dict can contain keys like 'prefill_ms'/'decode_ms'/'end2end_ms'
        return pred_text, {"end2end_ms": 123.4}
```

> **Integration Key Points**:
>
> 1.  Classes in `dataset.py / metric.py / model.py` must adhere to the interfaces defined in `scripts.abstract` (`BaseDataset` / `BaseMetric` / `BaseModel`).
> 2.  New classes can be selected in `config/*.yaml`. To **temporarily override** the YAML configuration, use CLI parameters (see next section).

-----

## 3\) Running Evaluation with `tools/mme_run_all.py`

The script performs the following actions:

1.  Automatically discovers **subtasks** (`<MME_Benchmark>/<SubtaskName>/` subdirectories) or runs based on the `--only` specified list.
2.  Builds/normalizes annotations for each subtask (prioritizing `tools/mme_build_ann.py`, falling back to `tools/mme_ann_from_txtpairs.py`).
3.  Calls the unified `scripts/evalkit.py` for inference and scoring.
4.  Summarizes results into `mme_summary.csv` and per-subtask `results.jsonl`.

### Minimal Example (Run only the OCR subtask)

```bash
python tools/mme_run_all.py   --cfg config/sparsevlm_mme.yaml   --mme_root /home/w1nd519994824/project/VisionZip-exp/datasets/mme/MME_Benchmark_release_version/MME_Benchmark   --out_root eval_results/mme_eval_results   --only OCR
```

### Key Parameter Descriptions

  - `--cfg`: YAML configuration file (determines default dataset / model / metrics / decoding parameters, etc.).
  - `--mme_root`: The MME root directory (pointing to `.../MME_Benchmark`).
  - `--out_root`: Output root directory (the script will create subdirectories within this path:
      - `ann/`: Standardized annotations (JSON) for each subtask.
      - `results/`: Prediction and statistics outputs for each subtask.
      - `mme_summary.csv`: Summary table across all subtasks).
  - `--only`: Optional, restricts the evaluation to specific subtasks; supports **multiple subtask names separated by spaces**, e.g.: `--only OCR color counting`.

### Advanced Overrides (Optional)

The following parameters can **temporarily override** the YAML file via the command line:

  - `--dataset [vqa|mme|pope]`
  - `--model_type [llava_vzip|sparsevlm]`
  - `--model_path /path/or/hf-id`
  - `--temperature FLOAT`, `--max_new_tokens INT`
  - `--warmup INT`, `--seed INT`, `--limit INT`
  - `--dominant INT`, `--contextual INT`, `--retained_tokens INT` (VisionZip related)
  - `--conv_mode STR`

> **Output Expectation**: After execution, a brief summary will be printed to the terminal, and `mme_summary.csv` will be generated under the `--out_root` path.

-----

## Frequently Asked Questions (FAQ)

  - **Error prompts require HF login?** Run `huggingface-cli login` and try again.
  - **Want to debug a subset?** Use `--only` to specify 1\~N subtasks, or add `--limit` to load only the first N samples.
  - **How to add/replace metrics?** Implement a subclass of `BaseMetric` in `metric.py` and switch in the YAML, or directly use CLI overrides.
  - **MME annotation schema inconsistency?** `mme_run_all.py` attempts to **flatten paired annotations** (and injects `image_id/pair`) to enable ACC+ calculation.

-----

  ## SparseZip Vision Token Compression

  We added an experimental vision token compressor called **SparseZip** that performs dynamic dominant token selection (adaptive K) and hierarchical contextual merging to reduce the number of image patch tokens passed to the LLM while preserving salient content.

  Key ideas (see detailed doc for equations and implementation details):

  - Hybrid scoring per patch: attention + entropy + mutual-information proxy (weighted by YAML `alphas`).
  - Dynamic-K selection: `K = round(log(var(scores)+eps) + c)` (bounded by `k_min`/`k_max`).
  - Hierarchical merging of non-dominant patches into fewer contextual tokens (k-means init + optional agglomerative refinement).
  - Optional cross-attn fusion and multi-layer gating (API-ready; cross-attn off by default).

  Code entry points:

  - Compressor: `utils/sparsezip.py` (class `VisionZipCompressor`).
  - Model patch: `scripts/model.py` (patched `CLIPVisionTower.forward` via `_sparsezip_forward`).
  - YAML examples: `config/sparsezip_mme.yaml`, `config/sparsezip_pope.yaml` (`model.sparsezip` section).

  To enable SparseZip, use one of the `config/sparsezip_*.yaml` files (or add a `sparsezip:` section under `model:` in your own YAML). Example override flags still work for legacy `dominant/contextual`, but when `model.sparsezip.dynamic_k: true` the compressor will adapt K per image.

  Minimal run example (MME OCR only):

  ```bash
  python tools/mme_run_all.py \
    --mme_root /path/to/MME_Benchmark \
    --out_root ./runs/mme_sparsezip \
    --cfg config/sparsezip_mme.yaml \
    --only OCR
  ```

  Full documentation: see `docs/README_SPARSEZIP.md`.

  -----

## Directory and Artifact Examples

```
repo/
  tools/
    mme_run_all.py
    mme_build_ann.py
    mme_ann_from_txtpairs.py
  scripts/
    evalkit.py
    dataset_download.py
    abstract.py
    timer.py
    dataset.py     # Customizable
    metric.py      # Customizable
    model.py       # Customizable
  config/
    sparsevlm_mme.yaml
  

datasets/
  mme/MME_Benchmark_release_version/MME_Benchmark/...

eval_results/
  mme_eval_results/
    ann/
      OCR.json
      ...
    results/
      OCR/
        results.jsonl
        scores.json
        logs.jsonl
      ...
    mme_summary.csv
```
