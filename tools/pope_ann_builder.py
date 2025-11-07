#!/usr/bin/env python3
# tools/pope_ann_builder.py
import os, json, argparse
from pathlib import Path

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def load_annotation(path: Path):
    """加载 POPE 原始标注文件"""
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
    else:
        raise ValueError(f"Unknown annotation format: {path}")

def ensure_suffix(q: str) -> str:
    """确保问题以 'Please answer yes or no.' 结尾"""
    q = (q or "").strip()
    suff = "Please answer yes or no."
    return q if q.lower().endswith(suff.lower()) else (q.rstrip() + " " + suff)

def resolve_image(img_root: Path, image_name: str) -> Path:
    """解析图片路径"""
    p = Path(image_name)
    
    # 如果是绝对路径且存在
    if p.is_absolute() and p.exists():
        return p
    
    # 尝试直接拼接
    cand = img_root / image_name
    if cand.exists():
        return cand
    
    # 尝试不同扩展名
    stem = Path(image_name).stem
    for ext in VALID_IMG_EXT:
        c = img_root / f"{stem}{ext}"
        if c.exists():
            return c
    
    # 全局搜索（慢但稳）
    hits = list(img_root.rglob(f"{stem}*"))
    if hits:
        return hits[0]
    
    raise FileNotFoundError(f"Image not found for {image_name} under {img_root}")

def build_pope_ann(records, img_root: Path, variant: str = "random"):
    """
    构建 POPE 标准格式的标注
    输入格式示例:
    {
      "question_id": 0,
      "image": "COCO_val2014_000000000042.jpg",
      "text": "Is there a fork in the image?",
      "label": "no"
    }
    """
    items = []
    for idx, rec in enumerate(records):
        # 兼容不同字段命名
        qid = rec.get("question_id", idx)
        img = rec.get("image") or rec.get("image_path") or rec.get("image_id")
        question = rec.get("text") or rec.get("question") or rec.get("prompt")
        label = (rec.get("label") or rec.get("answer", "")).strip().lower()
        
        if img is None:
            print(f"[warn] Skip record {idx}: no image field")
            continue
        
        if question is None:
            print(f"[warn] Skip record {idx}: no question field")
            continue
        
        if label not in ("yes", "no"):
            print(f"[warn] Skip record {idx}: invalid label '{label}'")
            continue
        
        try:
            img_path = resolve_image(img_root, img)
        except FileNotFoundError as e:
            print(f"[warn] {e}")
            continue
        
        items.append({
            "question_id": str(qid),
            "image_path": str(img_path),
            "question": ensure_suffix(question),
            "label": label,
            "variant": variant  # random/popular/adversarial
        })
    
    return items

def main():
    ap = argparse.ArgumentParser(description="Build POPE annotation file")
    ap.add_argument("--pope_root", required=True, 
                    help="POPE dataset root directory")
    ap.add_argument("--variant", required=True, 
                    choices=["random", "popular", "adversarial"],
                    help="POPE variant: random/popular/adversarial")
    ap.add_argument("--img_root", required=True,
                    help="COCO image root directory (e.g., val2014/)")
    ap.add_argument("--out", required=True, 
                    help="Output annotation JSON path")
    args = ap.parse_args()

    pope_root = Path(args.pope_root).resolve()
    img_root = Path(args.img_root).resolve()
    
    # POPE 标注文件通常命名为: coco_pope_random.json / coco_pope_popular.json 等
    ann_file = pope_root / f"coco_pope_{args.variant}.json"
    if not ann_file.exists():
        # 尝试其他可能的命名
        alternatives = [
            pope_root / f"pope_{args.variant}.json",
            pope_root / f"{args.variant}.json",
            pope_root / f"COCO_pope_{args.variant}.json",
        ]
        for alt in alternatives:
            if alt.exists():
                ann_file = alt
                break
        else:
            raise FileNotFoundError(f"POPE annotation not found: {ann_file}")
    
    print(f"[build_ann] Using annotation: {ann_file}")
    
    records = load_annotation(ann_file)
    items = build_pope_ann(records, img_root, variant=args.variant)
    
    print(f"[build_ann] Built {len(items)} items for variant '{args.variant}'")
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    
    print(f"[build_ann] Saved to {out_path}")

if __name__ == "__main__":
    main()