#!/usr/bin/env python3
import os, json, csv, subprocess, sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

# ===== Default Configuration (Modify as needed) =====
MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DOMINANT = 54
CONTEXTUAL = 10
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 16
WARMUP = 5

QA_DIRNAME = "questions_answers_YN"  # Common MME structure (two TXT lines: positive/negative question)
VALID_IMG_EXT = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

# ===== Paths to dependent scripts (Consistent with the scripts you created earlier) =====
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
TXT_BUILDER = THIS_DIR / "mme_ann_from_txtpairs.py"  # Script to be created in section 2
GENERIC_BUILDER = THIS_DIR / "mme_ann_builder.py"    # Your previous generic builder (if available)
EVALKIT = REPO_ROOT / "evalkit.py"                   # Your evaluation framework script

def has_txt_pairs(subtask_dir: Path) -> bool:
    if any(subtask_dir.rglob("questions_answers_YN/*.txt")):
        return True
    return any(subtask_dir.rglob("*.txt"))

def has_generic_ann(subtask_dir: Path) -> bool:
    # Roughly check for the existence of json/jsonl/csv/tsv annotation files
    pats = ["*.json", "*.jsonl", "*.csv", "*.tsv", "*.txt"]
    for p in pats:
        if any(subtask_dir.glob(p)):
            return True
        if any(subtask_dir.rglob(p)):  # Deep search
            return True
    return False
def build_ann_for_subtask(mme_root: Path, subtask: str, out_json: Path):
    subtask_dir = mme_root / subtask
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if has_txt_pairs(subtask_dir):
        qa_glob = "questions_answers_YN/*.txt" if any(subtask_dir.rglob("questions_answers_YN/*.txt")) else "**/*.txt"
        cmd = [
            sys.executable, str(TXT_BUILDER),
            "--subtask_dir", str(subtask_dir),
            "--qa_glob", qa_glob,
            "--out", str(out_json),
        ]
        print("[build][txtpairs]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return True

    if GENERIC_BUILDER.exists() and has_generic_ann(subtask_dir):
        cmd = [
            sys.executable, str(GENERIC_BUILDER),
            "--mme_root", str(mme_root),
            "--subtask", subtask,
            "--out", str(out_json),
        ]
        print("[build][generic]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return True

    print(f"[warn] No recognizable annotations for {subtask_dir}")
    return False


def resolve_evalkit(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"evalkit not found: {p}")
        return p

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent
    candidates = [
        repo_root / "evalkit.py",
        repo_root / "scripts" / "evalkit.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "evalkit.py not found. Use --evalkit to specify its absolute path"
    )
    
def run_eval_for_subtask(ann_path: Path, out_dir: Path, evalkit_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(evalkit_path),
        "--dataset", "mme",
        "--model_path", MODEL_PATH,
        "--ann_path", str(ann_path),
        "--output_dir", str(out_dir),
        "--dominant", str(DOMINANT),
        "--contextual", str(CONTEXTUAL),
        "--temperature", str(TEMPERATURE),
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--warmup", str(WARMUP),
    ]
    print("[eval]", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(REPO_ROOT)}:{env.get('PYTHONPATH','')}"
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env) 

def read_summary_csv(csv_path: Path) -> dict:
    res = {}
    if not csv_path.exists():  
        return res
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        for k, v in r:
            # Handle conversion to number or string
            try:
                v = float(v)
            except:
                pass
            res[k] = v
    return res

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mme_root", required=True,
                    help="Root directory of MME_Benchmark (containing subtask directories)")
    ap.add_argument("--out_root", required=True,
                    help="Output root directory (for saving annotations and evaluation results)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only evaluate specified subtask names (can be multiple)")
    ap.add_argument("--evalkit", default=None, help="evalkit.py absolute path (default: auto-detect)")
    args = ap.parse_args()

    mme_root = Path(args.mme_root).resolve()
    out_root = Path(args.out_root).resolve()
    ann_dir = out_root / "ann"
    results_root = out_root / "results"
    evalkit_path = resolve_evalkit(args.evalkit)

    # Automatically discover subtasks (first-level subdirectories)
    subtasks = [p.name for p in mme_root.iterdir() if p.is_dir()]
    subtasks.sort()

    if args.only:
        subtasks = [s for s in subtasks if s in set(args.only)]
        print("[info] Filtered subtasks:", subtasks)

    # 1) Build annotations and run evaluation
    done = []
    for s in subtasks:
        print(f"\n=== Subtask: {s} ===")
        ann_path = ann_dir / f"ann_mme_{s}.json"
        ok = build_ann_for_subtask(mme_root, s, ann_path)
        if not ok:
            print(f"[skip] {s} (no annotations found)")
            continue
        out_dir = results_root / f"outputs_{s}"
        run_eval_for_subtask(ann_path, out_dir, evalkit_path)
        done.append(s)

    # 2) Summarize results
    print("\n=== Summarizing ===")
    summary_rows = []
    for s in done:
        summ = read_summary_csv(results_root / f"outputs_{s}" / "summary.csv")
        row = {"subtask": s}
        # Key metrics: mme_acc / mme_acc_plus
        for k in ["mme_acc", "mme_acc_plus",
                  "end2end_ms_avg", "end2end_ms_p50", "end2end_ms_p95",
                  "decode_ms_avg", "prefill_ms_avg", "decode_tok_per_s"]:
            if k in summ:
                row[k] = summ[k]
        summary_rows.append(row)

    # Write the master summary table
    summary_csv = out_root / "mme_summary.csv"
    keys = sorted({k for row in summary_rows for k in row.keys()})
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[ok] Summary saved to: {summary_csv}")

    # Also output a brief summary log
    def safe(v): 
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
    for row in summary_rows:
        print(" -", row["subtask"],
              "ACC=", safe(row.get("mme_acc","")),
              "ACC+=", safe(row.get("mme_acc_plus","")),
              "e2e_avg(ms)=", safe(row.get("end2end_ms_avg","")),
              "tok/s=", safe(row.get("decode_tok_per_s","")))

if __name__ == "__main__":
    main()