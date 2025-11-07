#!/usr/bin/env python3
# tools/pope_run_all.py
import os, json, csv, subprocess, sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

# ===== 默认配置 (根据需要修改) =====
MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DOMINANT = 54
CONTEXTUAL = 10
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 16
WARMUP = 5

# ===== 依赖脚本路径 =====
ANN_BUILDER = THIS_DIR / "pope_ann_builder.py"
EVALKIT = REPO_ROOT / "evalkit.py"

# POPE 的三个变体
POPE_VARIANTS = ["random", "popular", "adversarial"]

def resolve_evalkit(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"evalkit not found: {p}")
        return p

    candidates = [
        REPO_ROOT / "evalkit.py",
        REPO_ROOT / "scripts" / "evalkit.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "evalkit.py not found. Use --evalkit to specify its absolute path"
    )

def build_ann_for_variant(pope_root: Path, img_root: Path, 
                          variant: str, out_json: Path):
    """为指定的 POPE 变体构建标注文件"""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, str(ANN_BUILDER),
        "--pope_root", str(pope_root),
        "--variant", variant,
        "--img_root", str(img_root),
        "--out", str(out_json),
    ]
    print("[build]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return True

def run_eval_for_variant(ann_path: Path, out_dir: Path, 
                         evalkit_path: Path, model_path: str):
    """运行评估"""
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(evalkit_path),
        "--dataset", "pope",
        "--model_path", model_path,
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
    """读取 summary.csv"""
    res = {}
    if not csv_path.exists():
        return res
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        for k, v in r:
            try:
                v = float(v)
            except:
                pass
            res[k] = v
    return res

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run POPE evaluation for all variants")
    ap.add_argument("--pope_root", required=True,
                    help="POPE dataset root directory (containing annotation files)")
    ap.add_argument("--img_root", required=True,
                    help="COCO image root directory (e.g., val2014/)")
    ap.add_argument("--out_root", required=True,
                    help="Output root directory")
    ap.add_argument("--model_path", default=MODEL_PATH,
                    help=f"Model path (default: {MODEL_PATH})")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only evaluate specified variants (random/popular/adversarial)")
    ap.add_argument("--evalkit", default=None,
                    help="evalkit.py absolute path (default: auto-detect)")
    args = ap.parse_args()

    pope_root = Path(args.pope_root).resolve()
    img_root = Path(args.img_root).resolve()
    out_root = Path(args.out_root).resolve()
    ann_dir = out_root / "ann"
    results_root = out_root / "results"
    evalkit_path = resolve_evalkit(args.evalkit)

    # 选择要评估的变体
    variants = POPE_VARIANTS
    if args.only:
        variants = [v for v in variants if v in set(args.only)]
        print("[info] Filtered variants:", variants)

    # 1) 构建标注并运行评估
    done = []
    for variant in variants:
        print(f"\n=== POPE Variant: {variant} ===")
        ann_path = ann_dir / f"ann_pope_{variant}.json"
        
        try:
            build_ann_for_variant(pope_root, img_root, variant, ann_path)
        except Exception as e:
            print(f"[skip] {variant} (build failed: {e})")
            continue
        
        out_dir = results_root / f"outputs_{variant}"
        try:
            run_eval_for_variant(ann_path, out_dir, evalkit_path, args.model_path)
            done.append(variant)
        except Exception as e:
            print(f"[error] {variant} evaluation failed: {e}")

    # 2) 汇总结果
    print("\n=== Summarizing ===")
    summary_rows = []
    for variant in done:
        summ = read_summary_csv(results_root / f"outputs_{variant}" / "summary.csv")
        row = {"variant": variant}
        
        # 关键指标: accuracy, precision, recall, f1, latency
        for k in ["pope_acc", "pope_precision", "pope_recall", "pope_f1",
                  "end2end_ms_avg", "end2end_ms_p50", "end2end_ms_p95",
                  "decode_ms_avg", "decode_tok_per_s"]:
            if k in summ:
                row[k] = summ[k]
        summary_rows.append(row)

    # 写入总结表
    summary_csv = out_root / "pope_summary.csv"
    if summary_rows:
        keys = sorted({k for row in summary_rows for k in row.keys()})
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[ok] Summary saved to: {summary_csv}")

        # 输出简要日志
        def safe(v): 
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        
        print("\n=== Results ===")
        for row in summary_rows:
            print(f" - {row['variant']}:")
            print(f"     Acc={safe(row.get('pope_acc',''))}, "
                  f"P={safe(row.get('pope_precision',''))}, "
                  f"R={safe(row.get('pope_recall',''))}, "
                  f"F1={safe(row.get('pope_f1',''))}")
            print(f"     e2e_avg(ms)={safe(row.get('end2end_ms_avg',''))}, "
                  f"tok/s={safe(row.get('decode_tok_per_s',''))}")
    else:
        print("[warn] No results to summarize")

if __name__ == "__main__":
    main()