#!/usr/bin/env python3
# tools/mme_run_all.py
import os, sys, csv, json, subprocess
from pathlib import Path

# ---- repo paths ----
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

DEFAULT_EVALKIT = REPO_ROOT / "scripts" / "evalkit.py"

# ---- bring shared metrics into this runner ----
# These are used as the canonical metric classes & to decide which keys to collect in summary
try:
    from scripts.metric import MMEAcc, MMEAccPlus, DelayStats
except Exception as e:
    # Fall back gracefully; you can tighten this if you prefer hard failure
    print("[warn] Failed to import scripts.metric:", e)
    MMEAcc = MMEAccPlus = DelayStats = None

# Map dataset -> which metric keys we expect to aggregate in CSV summaries
# NOTE: keys match compute() outputs in your metric.py
METRIC_KEYS_BY_DATASET = {
    "mme": [
        # MMEAcc
        "mme_acc", "mme_count_q",
        # MMEAccPlus
        "mme_acc_plus", "mme_count_img",
        # DelayStats (common)
        "end2end_ms_avg", "end2end_ms_p50", "end2end_ms_p95",
        "decode_ms_avg", "decode_tok_per_s",
    ],
    # Other datasets can add their own metric keys here if needed
    # "vqa": [...],
    # "pope": [...],
}

# ---- annotation builders ----
PRIMARY_BUILDER_CANDIDATES = [
    THIS_DIR / "mme_build_ann.py",   # your current primary builder
    THIS_DIR / "mme_ann_builder.py", # legacy/alternate
]
TXT_PAIRS_BUILDER_CANDIDATES = [
    THIS_DIR / "mme_ann_from_txtpairs.py", # 2-line txt fallback
]

# ---------------- utils ----------------
def resolve_first_exist(cands):
    for p in cands:
        if p.exists():
            return p
    return None

def resolve_required(path_str: str | None, fallback_candidates: list[Path], hint: str) -> Path:
    if path_str:
        p = Path(path_str).resolve()
        if not p.exists():
            raise FileNotFoundError(f"{hint} not found: {p}")
        return p
    p = resolve_first_exist(fallback_candidates)
    if p is None:
        raise FileNotFoundError(f"{hint} not found in candidates: {fallback_candidates}")
    return p

def discover_subtasks(mme_root: Path) -> list[str]:
    return sorted([p.name for p in mme_root.iterdir() if p.is_dir()])

def _run(cmd, cwd=None, env=None, capture=False, check=True):
    if capture:
        return subprocess.run(cmd, cwd=cwd, env=env, check=check, text=True, capture_output=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=check)

# ------------- build annotations -------------
def build_ann_primary(primary_builder: Path, mme_root: Path, subtask: str, out_json: Path) -> bool:
    cmd = [
        sys.executable, str(primary_builder),
        "--mme_root", str(mme_root),
        "--subtask", subtask,
        "--out", str(out_json),
    ]
    print("[build-primary]", " ".join(cmd))
    try:
        _run(cmd, capture=False, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[build-primary] failed rc={e.returncode}")
        return False

def build_ann_txtpairs(txtpairs_builder: Path, subtask_dir: Path, out_json: Path) -> bool:
    cmd = [
        sys.executable, str(txtpairs_builder),
        "--subtask_dir", str(subtask_dir),
        "--qa_glob", "**/*.txt",
        "--out", str(out_json),
    ]
    print("[build-txtpairs]", " ".join(cmd))
    try:
        _run(cmd, capture=False, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[build-txtpairs] failed rc={e.returncode}")
        return False

def build_ann_for_subtask(primary_builder: Path | None,
                          txtpairs_builder: Path | None,
                          mme_root: Path, subtask: str, out_json: Path):
    """Try structured builder first, fallback to txt-pairs."""
    subtask_dir = mme_root / subtask
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if primary_builder is not None and build_ann_primary(primary_builder, mme_root, subtask, out_json):
        return
    if txtpairs_builder is not None and build_ann_txtpairs(txtpairs_builder, subtask_dir, out_json):
        return
    raise RuntimeError(f"Both builders failed for subtask '{subtask}'")

# ------------- flatten pairs -> per-question -------------
def _append_if_set(cmd: list[str], flag: str, value):
    if value is not None:
        cmd += [flag, str(value)]

def _ensure_suffix(q: str) -> str:
    if not q:
        return q
    suff = "Please answer yes or no."
    return q if q.lower().strip().endswith(suff.lower()) else (q.rstrip() + " " + suff)

def try_flatten_mme_ann(ann_path: Path, subtask_name: str):
    """
    If the built ann is in pair schema (q_pos/q_neg), rewrite in-place to per-question schema.
    Also inject meta for MMEAccPlus: meta={'image_id','pair'}
    """
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[flatten] skip (read failed): {ann_path} -> {e}")
        return

    if not isinstance(data, list) or not data:
        print(f"[flatten] skip (empty/non-list): {ann_path}")
        return

    sample = data[0]
    if "question_id" in sample:
        print(f"[flatten] already flat: {ann_path}")
        return

    if not all(k in sample for k in ("image_path", "q_pos", "q_neg")):
        print(f"[flatten] skip (not pair schema): {ann_path}")
        return

    flat = []
    for rec in data:
        img_path = rec.get("image_path")
        base_id = rec.get("image_id") or Path(img_path).stem

        qpos = _ensure_suffix(rec.get("q_pos") or "")
        qneg = _ensure_suffix(rec.get("q_neg") or "")

        if qpos:
            flat.append({
                "question_id": f"{base_id}_{subtask_name}_pos",
                "image_path": img_path,
                "question": qpos,
                "prompt": qpos,
                "answers": ["yes"],
                # critical for MMEAccPlus:
                "image_id": base_id,
                "pair": "pos",
                "meta": {"image_id": base_id, "pair": "pos"},
            })
        if qneg:
            flat.append({
                "question_id": f"{base_id}_{subtask_name}_neg",
                "image_path": img_path,
                "question": qneg,
                "prompt": qneg,
                "answers": ["no"],
                "image_id": base_id,
                "pair": "neg",
                "meta": {"image_id": base_id, "pair": "neg"},
            })

    if not flat:
        print(f"[flatten] produced 0 items, keep original: {ann_path}")
        return

    ann_path.write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[flatten] flattened -> {ann_path} ({len(flat)} items, with meta.image_id/pair)")

# ------------- call evalkit -------------
def run_eval_for_subtask(
    evalkit: Path,
    cfg_yaml: Path,
    ann_path: Path,
    out_dir: Path,
    # YAML-first; all following args only override when not None
    dataset=None, model_type=None, model_path=None,
    temperature=None, max_new_tokens=None, warmup=None, seed=None, limit=None,
    dominant=None, contextual=None, retained_tokens=None, conv_mode=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(evalkit), "--cfg", str(cfg_yaml)]

    # per-subtask must override:
    _append_if_set(cmd, "--ann_path", str(ann_path))
    _append_if_set(cmd, "--output_dir", str(out_dir))

    # optional overrides:
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
    env["PYTHONPATH"] = f"{str(REPO_ROOT)}:{env.get('PYTHONPATH','')}"
    _run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

# ------------- summarize -------------
def read_summary_csv(csv_path: Path) -> dict:
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
    # Prefer registry; fallback to a safe default
    return METRIC_KEYS_BY_DATASET.get(dataset, [
        "end2end_ms_avg","end2end_ms_p50","end2end_ms_p95","decode_ms_avg","decode_tok_per_s"
    ])

# ------------- main -------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run MME across subtasks (YAML-first, with shared metrics imported)")
    ap.add_argument("--mme_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="e.g., --only OCR color")

    ap.add_argument("--evalkit", default=None, help="Path to scripts/evalkit.py; default=auto")
    ap.add_argument("--primary_builder", default=None, help="Path to mme_build_ann.py / mme_ann_builder.py; default=auto")
    ap.add_argument("--txtpairs_builder", default=None, help="Path to mme_ann_from_txtpairs.py; default=auto")

    # YAML (required)
    ap.add_argument("--cfg", type=str, required=True, help="config/xxx.yaml")

    # Optional overrides (None = do not override YAML)
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

    mme_root = Path(args.mme_root).resolve()
    out_root = Path(args.out_root).resolve()
    ann_dir = out_root / "ann"
    results_root = out_root / "results"

    evalkit = resolve_required(args.evalkit, [DEFAULT_EVALKIT], "evalkit")

    # builders
    primary_builder = Path(args.primary_builder).resolve() if args.primary_builder else resolve_first_exist(PRIMARY_BUILDER_CANDIDATES)
    if primary_builder and not primary_builder.exists():
        primary_builder = None
    txtpairs_builder = Path(args.txtpairs_builder).resolve() if args.txtpairs_builder else resolve_first_exist(TXT_PAIRS_BUILDER_CANDIDATES)
    if txtpairs_builder and not txtpairs_builder.exists():
        txtpairs_builder = None

    cfg_yaml = Path(args.cfg).resolve()
    subtasks = args.only or discover_subtasks(mme_root)
    print("[info] subtasks:", subtasks)

    # If dataset override not provided, we don't know which key set to use ahead of time.
    # We'll assume 'mme' since this runner is for MME; but we also pass --dataset when provided.
    dataset_for_summary = args.dataset or "mme"

    done = []
    for sub in subtasks:
        print(f"\n=== MME Subtask: {sub} ===")
        ann_path = ann_dir / f"ann_mme_{sub}.json"
        try:
            build_ann_for_subtask(primary_builder, txtpairs_builder, mme_root, sub, ann_path)
            # flatten (and inject meta.image_id/pair) so MMEAccPlus can work
            try_flatten_mme_ann(ann_path, subtask_name=sub)
        except Exception as e:
            print(f"[skip] {sub} (build failed: {e})")
            continue

        out_dir = results_root / f"outputs_{sub}"
        try:
            run_eval_for_subtask(
                evalkit=evalkit,
                cfg_yaml=cfg_yaml,
                ann_path=ann_path,
                out_dir=out_dir,
                dataset=args.dataset,  # keep None to defer to YAML unless you override
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
            done.append(sub)
        except subprocess.CalledProcessError as e:
            print(f"[error] {sub} evaluation failed: returncode={e.returncode}")
        except Exception as e:
            print(f"[error] {sub} evaluation failed: {e}")

    # ---- summarize ----
    print("\n=== Summarizing ===")
    wanted_keys = keys_for_dataset(dataset_for_summary)
    summary_rows = []
    for sub in done:
        summ = read_summary_csv(results_root / f"outputs_{sub}" / "summary.csv")
        row = {"subtask": sub}
        for k in wanted_keys:
            if k in summ:
                row[k] = summ[k]
        summary_rows.append(row)

    summary_csv = out_root / "mme_summary.csv"
    if summary_rows:
        keys = sorted({k for r in summary_rows for k in r.keys()})
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[ok] Summary saved to: {summary_csv}")
        print("\n=== Results (brief) ===")
        for r in summary_rows:
            printable = ", ".join([f"{k}={r[k]}" for k in keys if k != "subtask" and k in r])
            print(f" - {r['subtask']}: {printable}")
    else:
        print("[warn] No results to summarize")

if __name__ == "__main__":
    main()