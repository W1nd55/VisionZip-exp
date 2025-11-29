#!/usr/bin/env python3
# tools/docvqa_run_all.py
import os
import sys
import csv
from pathlib import Path
from typing import Any

# ---- Repository Paths ----
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

DEFAULT_EVALKIT = REPO_ROOT / "scripts" / "evalkit.py"

# ---- Import metrics to know expected keys (only for summary) ----
try:
    from scripts.metric import DocVQAANLS, ExactMatch, DelayStats  # noqa: F401
except Exception as e:
    print("[warn] Failed to import scripts.metric:", e)
    DocVQAANLS = ExactMatch = DelayStats = None

# Metric keys expected to be collected from summary.csv (corresponding to metric.py's compute() output)
METRIC_KEYS_BY_DATASET = {
    "docvqa": [
        "docvqa_anls",
        "exact_match",
        "docvqa_count",  # from DocVQAANLS
        "count",         # from ExactMatch
        "end2end_ms_avg",
        "end2end_ms_p50",
        "end2end_ms_p95",
        "decode_ms_avg",
        "decode_tok_per_s",
    ],
}

# ---------------- Utils ----------------
def resolve_first_exist(cands: list[Path]) -> Path | None:
    """Returns the first path in the candidates list that exists."""
    for p in cands:
        if p.exists():
            return p
    return None

def resolve_required(path_str: str | None, fallback_candidates: list[Path], hint: str) -> Path:
    """Resolves a path from CLI or fallbacks, raising FileNotFoundError if missing."""
    if path_str:
        p = Path(path_str).resolve()
        if not p.exists():
            raise FileNotFoundError(f"{hint} not found: {p}")
        return p
    p = resolve_first_exist(fallback_candidates)
    if p is None:
        raise FileNotFoundError(f"{hint} not found in candidates: {fallback_candidates}")
    return p

def _run(cmd: list[str], cwd=None, env=None, capture=False, check=True):
    """Wrapper for subprocess.run."""
    import subprocess
    if capture:
        return subprocess.run(cmd, cwd=cwd, env=env, check=check, text=True, capture_output=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=check)

def _append_if_set(cmd: list[str], flag: str, value: Any):
    """Appends a CLI flag and its value to the command if the value is not None."""
    if value is not None:
        cmd += [flag, str(value)]

def keys_for_dataset(dataset: str) -> list[str]:
    """Returns the list of expected metric keys for a given dataset."""
    return METRIC_KEYS_BY_DATASET.get(
        dataset,
        ["end2end_ms_avg", "end2end_ms_p50", "end2end_ms_p95", "decode_ms_avg", "decode_tok_per_s"],
    )

def read_summary_csv(csv_path: Path) -> dict:
    """Reads key/value pairs from a simple two-column summary CSV."""
    res = {}
    if not csv_path.exists():
        return res
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) != 2:
                continue
            k, v = row
            try:
                v = float(v)
            except Exception:
                pass
            res[k] = v
    return res

# ---------------- DocVQA helpers ----------------
def resolve_docvqa_ann(docvqa_root: Path, split: str) -> Path:
    """
    Resolve DocVQA annotation JSON for a given split (train/val/test).

    We try a small list of common names, in order:
      docvqa_{split}_v1.0_withQT.json
      docvqa_{split}_v1.0.json
      docvqa_{split}.json
      *{split}*withQT*.json
      *{split}*.json
    """
    cand_names = [
        f"docvqa_{split}_v1.0_withQT.json",
        f"docvqa_{split}_v1.0.json",
        f"docvqa_{split}.json",
    ]
    for name in cand_names:
        p = docvqa_root / name
        if p.exists():
            return p

    # fallback: fuzzy glob
    globs = [
        f"*{split}*withQT*.json",
        f"*{split}*.json",
    ]
    for pat in globs:
        for p in sorted(docvqa_root.glob(pat)):
            return p

    raise FileNotFoundError(f"DocVQA annotation JSON for split '{split}' not found under {docvqa_root}")

def run_eval_for_split(
    evalkit: Path,
    cfg_yaml: Path,
    ann_path: Path,
    out_dir: Path,
    # YAML-first; the following only override YAML if not None
    dataset=None,
    model_type=None,
    model_path=None,
    temperature=None,
    max_new_tokens=None,
    warmup=None,
    seed=None,
    limit=None,
    dominant=None,
    contextual=None,
    retained_tokens=None,
    conv_mode=None,
):
    """Runs the evaluation script (evalkit) for a single DocVQA split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(evalkit), "--cfg", str(cfg_yaml)]

    _append_if_set(cmd, "--ann_path", str(ann_path))
    _append_if_set(cmd, "--output_dir", str(out_dir))

    # Optional overrides (None means defer to YAML)
    _append_if_set(cmd, "--dataset", dataset)
    _append_if_set(cmd, "--model_type", model_type)
    _append_if_set(cmd, "--model_path", model_path)
    _append_if_set(cmd, "--temperature", temperature)
    _append_if_set(cmd, "--max_new_tokens", max_new_tokens)
    _append_if_set(cmd, "--warmup", warmup)
    _append_if_set(cmd, "--seed", seed)
    _append_if_set(cmd, "--limit", limit)
    _append_if_set(cmd, "--dominant", dominant)
    _append_if_set(cmd, "--contextual", contextual)
    _append_if_set(cmd, "--retained_tokens", retained_tokens)
    _append_if_set(cmd, "--conv_mode", conv_mode)

    print("[eval]", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(REPO_ROOT)}:{env.get('PYTHONPATH', '')}"
    _run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

# ---------------- Main ----------------
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Run DocVQA (Single-Page Document VQA) across splits (YAML-first)."
    )
    ap.add_argument(
        "--docvqa_root",
        required=True,
        help="DocVQA root directory (contains docvqa_*json and documents/)",
    )
    ap.add_argument(
        "--out_root",
        required=True,
        help="Output root directory (will contain results/ and docvqa_summary.csv)",
    )
    ap.add_argument(
        "--splits",
        nargs="*",
        default=["val"],
        help="Which splits to evaluate (any subset of: train val test). Default: val",
    )

    ap.add_argument(
        "--evalkit",
        default=None,
        help="Path to scripts/evalkit.py; default=auto-detect",
    )

    # YAML (Required)
    ap.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to the configuration YAML file, e.g., config/llava_docvqa.yaml",
    )

    # The following override YAML values only if set; default None=no override
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["vqa", "mme", "pope", "coco_caption", "docvqa"],
    )
    ap.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["llava_vzip", "sparsezip", "sparsevlm", "llava"],
    )
    ap.add_argument("--model_path", type=str, default=None)

    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--dominant", type=int, default=None)
    ap.add_argument("--contextual", type=int, default=None)
    ap.add_argument("--retained_tokens", type=int, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)

    args = ap.parse_args()

    docvqa_root = Path(args.docvqa_root).resolve()
    out_root = Path(args.out_root).resolve()
    results_root = out_root / "results"

    evalkit = resolve_required(args.evalkit, [DEFAULT_EVALKIT], "evalkit")
    cfg_yaml = Path(args.cfg).resolve()

    splits = args.splits or ["val"]
    print("[info] splits:", splits)

    # dataset name used to select summary keys
    dataset_for_summary = (args.dataset or "docvqa").lower()

    done = []
    for split in splits:
        split = split.lower()
        print(f"\n=== DocVQA Split: {split} ===")

        try:
            ann_path = resolve_docvqa_ann(docvqa_root, split)
        except Exception as e:
            print(f"[skip] split={split} (cannot resolve ann: {e})")
            continue

        out_dir = results_root / f"outputs_{split}"

        try:
            run_eval_for_split(
                evalkit=evalkit,
                cfg_yaml=cfg_yaml,
                ann_path=ann_path,
                out_dir=out_dir,
                dataset=(args.dataset or "docvqa"),
                model_type=args.model_type,
                model_path=args.model_path,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                warmup=args.warmup,
                seed=args.seed,
                limit=args.limit,
                dominant=args.dominant,
                contextual=args.contextual,
                retained_tokens=args.retained_tokens,
                conv_mode=args.conv_mode,
            )
            done.append(split)
        except Exception as e:
            print(f"[error] split={split} evaluation failed: {e}")

    # ---- Summarize ----
    print("\n=== Summarizing ===")
    wanted_keys = keys_for_dataset(dataset_for_summary)
    summary_rows = []

    for split in done:
        summ = read_summary_csv(results_root / f"outputs_{split}" / "summary.csv")
        row = {"split": split}
        for k in wanted_keys:
            if k in summ:
                row[k] = summ[k]
        summary_rows.append(row)

    summary_csv = out_root / "docvqa_summary.csv"
    if summary_rows:
        keys = sorted({k for r in summary_rows for k in r.keys()})
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[ok] Summary saved to: {summary_csv}")
        print("\n=== Results (brief) ===")
        for r in summary_rows:
            printable = ", ".join(
                [f"{k}={r[k]}" for k in keys if k != "split" and k in r]
            )
            print(f" - {r['split']}: {printable}")
    else:
        print("[warn] No results to summarize")

if __name__ == "__main__":
    main()