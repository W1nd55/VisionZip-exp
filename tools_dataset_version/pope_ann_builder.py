#!/usr/bin/env python3
# tools/pope_ann_builder.py
import os, json, argparse
from pathlib import Path

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def load_pope_json(path: Path):
    """加载 POPE 原始 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def ensure_suffix(q: str) -> str:
    """确保问题以 'Please answer yes or no.' 结尾"""
    if not q:
        return q
    q = q.strip()
    suff = "Please answer yes or no."
    if q.lower().endswith(suff.lower()):
        return q
    # 有些问题已经有 "Answer yes or no" 但措辞不同
    if "yes or no" in q.lower():
        return q
    return q.rstrip() + " " + suff

def resolve_image(img_root: Path, image_name: str) -> Path:
    """解析图片路径"""
    # POPE 的图片名通常是 COCO_val2014_000000xxxxxx.jpg
    p = Path(image_name)
    
    # 如果是绝对路径且存在
    if p.is_absolute() and p.exists():
        return p
    
    # 只有文件名，尝试在 img_root 下查找
    if not p.parent or p.parent == Path('.'):
        cand = img_root / image_name
        if cand.exists():
            return cand
        
        # 尝试不同扩展名
        stem = p.stem
        for ext in VALID_IMG_EXT:
            c = img_root / f"{stem}{ext}"
            if c.exists():
                return c
    
    # 相对路径，直接拼接
    cand = img_root / image_name
    if cand.exists():
        return cand
    
    raise FileNotFoundError(f"Image not found: {image_name} under {img_root}")

def build_pope_ann(pope_json_path: Path, img_root: Path, variant: str) -> list:
    """
    将 POPE 原始格式转换为评估格式
    
    POPE 原始格式:
    [
      {
        "question_id": 0,
        "image": "COCO_val2014_000000000042.jpg",
        "text": "Is there a fork in the image?",
        "label": "no"
      },
      ...
    ]
    
    目标格式:
    [
      {
        "question_id": "0",
        "image_path": "/full/path/to/image.jpg",
        "question": "Is there a fork in the image? Please answer yes or no.",
        "label": "no",
        "variant": "random"
      },
      ...
    ]
    """
    print(f"[build] Loading {pope_json_path}")
    data = load_pope_json(pope_json_path)
    print(f"[build] Loaded {len(data)} questions")
    
    items = []
    skipped = 0
    
    for idx, rec in enumerate(data):
        # 提取字段（兼容不同命名）
        qid = rec.get("question_id", idx)
        img = rec.get("image") or rec.get("image_path") or rec.get("image_id")
        question = rec.get("text") or rec.get("question") or rec.get("prompt")
        label = (rec.get("label") or rec.get("answer", "")).strip().lower()
        
        # 验证必需字段
        if not img:
            print(f"[warn] Skip record {idx}: no image field")
            skipped += 1
            continue
        
        if not question:
            print(f"[warn] Skip record {idx}: no question field")
            skipped += 1
            continue
        
        if label not in ("yes", "no"):
            print(f"[warn] Skip record {idx}: invalid label '{label}'")
            skipped += 1
            continue
        
        # 解析图片路径
        try:
            img_path = resolve_image(img_root, img)
        except FileNotFoundError as e:
            print(f"[warn] Skip record {idx}: {e}")
            skipped += 1
            continue
        
        # 构建条目
        items.append({
            "question_id": str(qid),
            "image_path": str(img_path),
            "question": ensure_suffix(question),
            "label": label,
            "variant": variant
        })
    
    print(f"[build] Built {len(items)} items, skipped {skipped}")
    return items

def main():
    ap = argparse.ArgumentParser(description="Build POPE annotation file for evaluation")
    ap.add_argument("--pope_root", required=True, 
                    help="POPE dataset root directory (containing coco_pope_*.json files)")
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
    
    # 查找 POPE 标注文件
    possible_names = [
        f"coco_pope_{args.variant}.json",
        f"pope_{args.variant}.json",
        f"{args.variant}.json",
        f"COCO_pope_{args.variant}.json",
    ]
    
    pope_json = None
    for name in possible_names:
        candidate = pope_root / name
        if candidate.exists():
            pope_json = candidate
            break
    
    if not pope_json:
        raise FileNotFoundError(
            f"POPE annotation not found. Tried: {possible_names}\n"
            f"in directory: {pope_root}"
        )
    
    print(f"[build] Using annotation: {pope_json}")
    print(f"[build] Image root: {img_root}")
    
    # 构建标注
    items = build_pope_ann(pope_json, img_root, variant=args.variant)
    
    # 保存
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    
    print(f"[build] Saved to {out_path}")

if __name__ == "__main__":
    main()