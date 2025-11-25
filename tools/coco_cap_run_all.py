#!/usr/bin/env python3
# tools/coco_cap_run_all.py
import os
import sys
import csv
from pathlib import Path
from typing import Any

# ---- Repository Paths ----
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

DEFAULT_EVALKIT = REPO_ROOT / "scripts" / "evalkit.py"

# ---- Import metrics just to know key names (optional, for sanity) ----
try:
    from scripts.metric import (
        CaptionBLEU,
        CaptionROUGEL,
        CaptionMETEOR,
        CaptionCIDEr,
        DelayStats,  # noqa: F401
    )
except Exception as e:
    print("[warn] Failed to import scripts.metric:", e)
    CaptionBLEU = CaptionROUGEL = CaptionMETEOR = CaptionCIDEr = DelayStats = None

# Metric keys expected to be collected from summary.csv
METRIC_KEYS_BY_DATASET = {
    "coco_caption": [
        "bleu1",
        "bleu2",
        "bleu3",
        "bleu4",
        "rouge_l",
        "meteor",
        "cider",
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


def _run(cmd: list[str], cwd=None, env=None, check=True):
    """Wrapper for subprocess.run."""
    import subprocess

    print("[eval]", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, check=check)


def _append_if_set(cmd: list[str], flag: str, value: Any):
    """Appends a CLI flag and its value to the command if the value is not None."""
    if value is not None:
        cmd += [flag, str(value)]


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


def keys_for_dataset(dataset: str) -> list[str]:
    """Returns the list of expected metric keys for a given dataset."""
    return METRIC_KEYS_BY_DATASET.get(
        dataset,
        [
            "end2end_ms_avg",
            "end2end_ms_p50",
            "end2end_ms_p95",
            "decode_ms_avg",
            "decode_tok_per_s",
        ],
    )


# ---------------- Main Logic ----------------
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Run COCO Caption evaluation (BLEU/METEOR/ROUGE-L/CIDEr) via evalkit"
    )
    # 必须：caption 标注 JSON（例如 datasets/coco/annotations/annotations/captions_val2014.json）
    ap.add_argument(
        "--ann_path",
        type=str,
        required=True,
        help="Path to COCO caption annotation JSON (e.g., captions_val2014.json)",
    )
    # 输出根目录
    ap.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root directory for this evaluation",
    )

    # evalkit & cfg
    ap.add_argument(
        "--evalkit",
        default=None,
        help="Path to scripts/evalkit.py; default=auto-detect",
    )
    ap.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to configuration YAML (e.g., config/xxx_coco_cap.yaml)",
    )

    # Dataset/model overrides（与 pope_run_all 保持一致）
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["coco_caption", "caption"],
        help="Dataset name override (default: coco_caption)",
    )
    ap.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["llava_vzip", "sparsevlm"],
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

    ann_path = Path(args.ann_path).resolve()
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {ann_path}")

    out_root = Path(args.out_root).resolve()
    results_dir = out_root / "results_coco_caption"
    results_dir.mkdir(parents=True, exist_ok=True)

    evalkit = resolve_required(args.evalkit, [DEFAULT_EVALKIT], "evalkit")
    cfg_yaml = Path(args.cfg).resolve()

    # ---- 1. 构造 evalkit 命令 ----
    cmd = [sys.executable, str(evalkit), "--cfg", str(cfg_yaml)]

    # 必传：ann_path & output_dir
    _append_if_set(cmd, "--ann_path", str(ann_path))
    _append_if_set(cmd, "--output_dir", str(results_dir))

    # Dataset 默认为 coco_caption
    dataset_name = (args.dataset or "coco_caption")
    _append_if_set(cmd, "--dataset", dataset_name)

    # 其它 override：
    _append_if_set(cmd, "--model_type", args.model_type)
    _append_if_set(cmd, "--model_path", args.model_path)
    _append_if_set(cmd, "--temperature", args.temperature)
    _append_if_set(cmd, "--max_new_tokens", args.max_new_tokens)
    _append_if_set(cmd, "--warmup", args.warmup)
    _append_if_set(cmd, "--seed", args.seed)
    _append_if_set(cmd, "--limit", args.limit)
    _append_if_set(cmd, "--dominant", args.dominant)
    _append_if_set(cmd, "--contextual", args.contextual)
    _append_if_set(cmd, "--retained_tokens", args.retained_tokens)
    _append_if_set(cmd, "--conv_mode", args.conv_mode)

    # ---- 2. 调 evalkit ----
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(REPO_ROOT)}:{env.get('PYTHONPATH', '')}"

    try:
        _run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    except Exception as e:
        print(f"[error] COCO Caption evaluation failed: {e}")
        raise

    # ---- 3. Summarize ----
    print("\n=== Summarizing COCO Caption Metrics ===")
    wanted_keys = keys_for_dataset(dataset_name.lower())

    summary_csv_in = results_dir / "summary.csv"
    summ = read_summary_csv(summary_csv_in)

    row = {}
    for k in wanted_keys:
        if k in summ:
            row[k] = summ[k]

    summary_csv_out = out_root / "coco_caption_summary.csv"
    if row:
        keys = ["metric", "value"]
        with open(summary_csv_out, "w", newline="") as f:
            w = csv.writer(f)
            for k in wanted_keys:
                if k in row:
                    w.writerow([k, row[k]])
        print(f"[ok] Summary saved to: {summary_csv_out}")
        print("=== Results (brief) ===")
        for k in wanted_keys:
            if k in row:
                print(f" - {k}: {row[k]:.4f}" if isinstance(row[k], float) else f" - {k}: {row[k]}")
    else:
        print("[warn] No recognized metrics found in summary.csv")

if __name__ == "__main__":
    main()