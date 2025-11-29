#!/usr/bin/env python3
# tools/pope_run_all.py
import os, sys, csv, json, subprocess
from pathlib import Path
from typing import Any

# ---- Repository Paths ----
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

DEFAULT_EVALKIT = REPO_ROOT / "scripts" / "evalkit.py"

# ---- Bring shared metrics into this runner (Only used for determining summary fields) ----
try:
    # Importing metrics to get access to their key names for summarization
    from scripts.metric import POPEAcc, POPEPrecisionRecallF1, DelayStats  # noqa: F401
except Exception as e:
    print("[warn] Failed to import scripts.metric:", e)
    POPEAcc = POPEPrecisionRecallF1 = DelayStats = None

# Metric keys expected to be collected from summary.csv (corresponding to metric.py's compute() output)
METRIC_KEYS_BY_DATASET = {
    "pope": [
        "pope_acc", "pope_precision", "pope_recall", "pope_f1",
        "end2end_ms_avg", "end2end_ms_p50", "end2end_ms_p95",
        "decode_ms_avg", "decode_tok_per_s",
    ],
}

# ---- Annotation Builder Candidates ----
ANN_BUILDER_CANDIDATES = [
    THIS_DIR / "pope_ann_builder.py",
]

POPE_VARIANTS = ["random", "popular", "adversarial"]

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

def _run(cmd: list[str], cwd=None, env=None, capture=False, check=True) -> subprocess.CompletedProcess:
    """Wrapper for subprocess.run."""
    if capture:
        return subprocess.run(cmd, cwd=cwd, env=env, check=check, text=True, capture_output=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=check)

def _append_if_set(cmd: list[str], flag: str, value: Any):
    """Appends a CLI flag and its value to the command if the value is not None."""
    if value is not None:
        cmd += [flag, str(value)]

# ------------- Build Annotations -------------
def build_ann_for_variant(ann_builder: Path, pope_root: Path, img_root: Path, variant: str, out_json: Path) -> bool:
    """Runs the annotation builder script for a specific POPE variant."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ann_builder),
        "--pope_root", str(pope_root),
        "--variant", variant,
        "--img_root", str(img_root),
        "--out", str(out_json),
    ]
    print("[build]", " ".join(cmd))
    try:
        _run(cmd, capture=False, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[build] {variant} failed rc={e.returncode}")
        return False

# ------------- Call Evalkit -------------
def run_eval_for_variant(
    evalkit: Path,
    cfg_yaml: Path,
    ann_path: Path,
    out_dir: Path,
    # YAML-first; the following only override YAML if not None
    dataset=None, model_type=None, model_path=None,
    temperature=None, max_new_tokens=None, warmup=None, seed=None, limit=None,
    dominant=None, contextual=None, retained_tokens=None, conv_mode=None,
):
    """Runs the evaluation script (evalkit) for a single variant."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(evalkit), "--cfg", str(cfg_yaml)]

    # Essential inputs/outputs specific to each variant must be passed
    _append_if_set(cmd, "--ann_path", str(ann_path))
    _append_if_set(cmd, "--output_dir", str(out_dir))

    # Optional overrides (None means defer to YAML)
    _append_if_set(cmd, "--dataset", dataset)              # Recommended to pass "pope" or set in YAML
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
    env["PYTHONPATH"] = f"{str(REPO_ROOT)}:{env.get('PYTHONPATH','')}"
    # Run the evaluation command from the repo root
    _run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

# ------------- Summarize -------------
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
                # Attempt to convert value to float
                v = float(v)
            except Exception:
                pass
            res[k] = v
    return res

def keys_for_dataset(dataset: str) -> list[str]:
    """Returns the list of expected metric keys for a given dataset."""
    return METRIC_KEYS_BY_DATASET.get(dataset, [
        "end2end_ms_avg","end2end_ms_p50","end2end_ms_p95","decode_ms_avg","decode_tok_per_s"
    ])

# ------------- Main Entry Point -------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run POPE across variants (YAML-first, with shared metrics imported)")
    ap.add_argument("--pope_root", required=True, help="POPE dataset root (annotations)")
    ap.add_argument("--img_root", required=True, help="COCO image root (e.g., val2014/)")
    ap.add_argument("--out_root", required=True, help="Output root directory")
    ap.add_argument("--only", nargs="*", default=None, help="Only evaluate specified variants, e.g., --only random adversarial")

    ap.add_argument("--evalkit", default=None, help="Path to scripts/evalkit.py; default=auto-detect")
    ap.add_argument("--ann_builder", default=None, help="Path to tools/pope_ann_builder.py; default=auto-detect")

    # YAML (Required)
    ap.add_argument("--cfg", type=str, required=True, help="Path to the configuration YAML file, e.g., config/xxx.yaml")

    # The following override YAML values only if set; default None=no override
    ap.add_argument("--dataset", type=str, default=None, choices=["vqa","mme","pope"])
    ap.add_argument("--model_type", type=str, default=None, choices=["llava_vzip","sparsevlm"])
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

    # Resolve paths
    pope_root = Path(args.pope_root).resolve()
    img_root = Path(args.img_root).resolve()
    out_root = Path(args.out_root).resolve()
    ann_dir = out_root / "ann"
    results_root = out_root / "results"

    evalkit = resolve_required(args.evalkit, [DEFAULT_EVALKIT], "evalkit")
    ann_builder = resolve_required(args.ann_builder, ANN_BUILDER_CANDIDATES, "pope_ann_builder")

    cfg_yaml = Path(args.cfg).resolve()

    variants = args.only or POPE_VARIANTS
    print("[info] variants:", variants)

    # Dataset name used to select summary keys (defaults to 'pope' for this runner)
    dataset_for_summary = (args.dataset or "pope").lower()

    done = []
    for variant in variants:
        print(f"\n=== POPE Variant: {variant} ===")
        ann_path = ann_dir / f"ann_pope_{variant}.json"

        # 1. Build annotations
        try:
            ok = build_ann_for_variant(ann_builder, pope_root, img_root, variant, ann_path)
            if not ok:
                print(f"[skip] {variant} (build failed)")
                continue
        except Exception as e:
            print(f"[skip] {variant} (build failed: {e})")
            continue

        # 2. Run evaluation
        out_dir = results_root / f"outputs_{variant}"
        try:
            # If --dataset is not explicitly provided, fix it to "pope" to prevent YAML misconfiguration
            run_eval_for_variant(
                evalkit=evalkit,
                cfg_yaml=cfg_yaml,
                ann_path=ann_path,
                out_dir=out_dir,
                dataset=(args.dataset or "pope"),
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
            done.append(variant)
        except subprocess.CalledProcessError as e:
            print(f"[error] {variant} evaluation failed: returncode={e.returncode}")
        except Exception as e:
            print(f"[error] {variant} evaluation failed: {e}")

    # ---- 3. Summarize results ----
    print("\n=== Summarizing ===")
    wanted_keys = keys_for_dataset(dataset_for_summary)
    summary_rows = []
    for variant in done:
        summ = read_summary_csv(results_root / f"outputs_{variant}" / "summary.csv")
        row = {"variant": variant}
        for k in wanted_keys:
            if k in summ:
                row[k] = summ[k]
        summary_rows.append(row)

    summary_csv = out_root / "pope_summary.csv"
    if summary_rows:
        keys = sorted({k for r in summary_rows for k in r.keys()})
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[ok] Summary saved to: {summary_csv}")
        print("\n=== Results (brief) ===")
        for r in summary_rows:
            printable = ", ".join([f"{k}={r[k]}" for k in keys if k != "variant" and k in r])
            print(f" - {r['variant']}: {printable}")
    else:
        print("[warn] No results to summarize")

if __name__ == "__main__":
    main()