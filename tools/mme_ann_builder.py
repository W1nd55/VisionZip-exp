# tools/mme_build_ann.py
import os, json, argparse, glob, csv
from pathlib import Path

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def find_annotation_file(subdir: Path):
    # 常见命名优先级
    candidates = [
        "qa.json", "questions.json", "annotation.json",
        "mme_qa.json", "pairs.json", "eval.json",
    ]
    for c in candidates:
        p = subdir / c
        if p.exists():
            return p

    # 兜底：扫 json/jsonl/csv/tsv
    js = list(subdir.glob("*.json"))
    if js: return js[0]
    jl = list(subdir.glob("*.jsonl"))
    if jl: return jl[0]
    cs = list(subdir.glob("*.csv"))
    if cs: return cs[0]
    ts = list(subdir.glob("*.tsv"))
    if ts: return ts[0]
    return None

def load_annotation(path: Path):
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif path.suffix.lower() == ".jsonl":
        out = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out
    elif path.suffix.lower() in (".csv", ".tsv"):
        delim = "," if path.suffix.lower()==".csv" else "\t"
        out = []
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=delim)
            for row in r:
                out.append(row)
        return out
    else:
        raise ValueError(f"Unknown annotation format: {path}")

def guess_fields(rec: dict):
    """
    尝试兼容不同字段命名，返回：
    image (str), q_pos (str), a_pos ('yes'), q_neg (str), a_neg ('no')
    允许记录里就是分开的两条（正/负各一条），由上层聚合；也允许一条里含正负成对。
    """
    # 常见字段名候选
    img_keys = ["image", "image_path", "img", "img_path", "image_name", "file_name"]
    qpos_keys = ["q_pos", "question_pos", "question_yes", "positive_question"]
    apos_keys = ["a_pos", "answer_pos", "answer_yes", "positive_answer"]
    qneg_keys = ["q_neg", "question_neg", "question_no", "negative_question"]
    aneg_keys = ["a_neg", "answer_neg", "answer_no", "negative_answer"]

    def first(keys):
        for k in keys:
            if k in rec:
                return rec[k]
        return None

    image = first(img_keys)
    q_pos = first(qpos_keys)
    a_pos = first(apos_keys)
    q_neg = first(qneg_keys)
    a_neg = first(aneg_keys)

    # 如果是单条 yes/no 记录，需要上层先聚合；这里直接返回原样
    return image, q_pos, a_pos, q_neg, a_neg

def build_pairs(records, img_root: Path):
    """
    尝试把 annotation 组装成 MME 的“每图两题（pos/neg）”结构。
    兼容两种来源：
      1) 每条记录里已经包含 q_pos/q_neg
      2) 每条只是单个 yes/no 问题，需要根据 image_id 归并为一对
    """
    # 尝试直接成对字段
    paired = []
    singles_by_image = {}

    for rec in records:
        image, q_pos, a_pos, q_neg, a_neg = guess_fields(rec)

        # 兜底：一些文件把图片名放在 'image_id' 里
        if image is None and "image_id" in rec:
            image = rec["image_id"]

        # 再兜底：question/answer 字段只有一条
        if q_pos is None and q_neg is None:
            q = rec.get("question") or rec.get("q") or rec.get("prompt")
            a = (rec.get("answer") or rec.get("a") or rec.get("label") or "").strip().lower()
            if image is None:
                # 有的把图片文件放 'image' / 'img'，上面已经尽力了
                raise ValueError(f"Cannot find image field in record: {rec}")
            singles_by_image.setdefault(image, []).append((q, a))
            continue

        if image is None:
            raise ValueError(f"Missing image field in paired record: {rec}")

        # 标准化 yes/no
        def yn(x, default):
            x = (x or default or "").strip().lower()
            return "yes" if x.startswith("y") else ("no" if x.startswith("n") else default)

        a_pos = yn(a_pos, "yes")
        a_neg = yn(a_neg, "no")

        paired.append({
            "image_id": Path(image).stem,
            "image_path": str(resolve_image(img_root, image)),
            "q_pos": ensure_suffix(q_pos, "Please answer yes or no."),
            "a_pos": a_pos,
            "q_neg": ensure_suffix(q_neg, "Please answer yes or no."),
            "a_neg": a_neg
        })

    # 处理单条 -> 成对（按 image 分两条：一条答案 yes，一条答案 no）
    for image, qa_list in singles_by_image.items():
        # 选出一条 yes 和一条 no（如果超过一条，取前两条）
        yes_q = next((q for q,a in qa_list if a in ("yes","y","1","true")), None)
        no_q  = next((q for q,a in qa_list if a in ("no","n","0","false")), None)
        if yes_q is None or no_q is None:
            # 数据不完整，跳过或抛错均可；这里选择跳过
            continue
        paired.append({
            "image_id": Path(image).stem,
            "image_path": str(resolve_image(img_root, image)),
            "q_pos": ensure_suffix(yes_q, "Please answer yes or no."),
            "a_pos": "yes",
            "q_neg": ensure_suffix(no_q,  "Please answer yes or no."),
            "a_neg": "no"
        })

    return paired

def ensure_suffix(q, suffix):
    if q is None:
        return None
    q = q.strip()
    suff = suffix.strip()
    return q if q.lower().endswith(suff.lower()) else (q.rstrip() + " " + suff)

def resolve_image(img_root: Path, image_name: str) -> Path:
    p = Path(image_name)
    if p.is_absolute() and p.exists():
        return p
    # 尝试直接拼接
    cand = img_root / image_name
    if cand.exists():
        return cand
    # 广播匹配
    stem = Path(image_name).stem
    for ext in VALID_IMG_EXT:
        c = img_root / f"{stem}{ext}"
        if c.exists():
            return c
    # 最后全量搜索（慢，但稳）
    hits = list(img_root.rglob(f"{stem}*"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Image not found for {image_name} under {img_root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mme_root", required=True, help="MME_Benchmark 目录（包含 existence/ color/ ... 子目录）")
    ap.add_argument("--subtask", required=True, help="子任务目录名，例如 existence / color / OCR ...")
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    args = ap.parse_args()

    subdir = Path(args.mme_root) / args.subtask
    if not subdir.exists():
        raise FileNotFoundError(subdir)

    ann_file = find_annotation_file(subdir)
    if ann_file is None:
        raise FileNotFoundError(f"No annotation file found in {subdir}. Please put qa.json / questions.json / annotation.json etc.")
    print(f"[build_ann] Using annotation: {ann_file}")

    records = load_annotation(ann_file)
    paired = build_pairs(records, img_root=subdir)
    print(f"[build_ann] Built {len(paired)} paired items")

    with open(args.out, "w") as f:
        json.dump(paired, f, ensure_ascii=False, indent=2)
    print(f"[build_ann] Saved to {args.out}")

if __name__ == "__main__":
    main()