#!/usr/bin/env python3
# tools/pope_ann_builder.py
import os, json, argparse
from pathlib import Path

# Valid image extensions for globbing/checking
VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def _norm_yesno(x):
    """Normalizes 'yes/no' answers to 'yes' or 'no'."""
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return "yes"
    if s in ("no", "n", "false", "0"):
        return "no"
    return None

def load_annotation(path: Path):
    """
    Loads the original POPE annotation file:
      - Tries standard JSON first (top level is list or {"data": list})
      - Falls back to parsing line by line as JSON Lines (JSONL)
    Handles both .json and .jsonl extensions.
    """
    p = Path(path)
    # Try standard JSON first
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            return obj["data"]
        # If structure is unexpected, proceed to JSONL attempt
    except json.JSONDecodeError:
        pass

    # Fallback: Parse line by line as JSONL
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"{p}: line {i} is not valid JSON. Head: {line[:120]}"
                ) from e
    return out

def ensure_suffix(q: str) -> str:
    """Ensures the question ends with 'Please answer yes or no.'"""
    q = (q or "").strip()
    suff = "Please answer yes or no."
    return q if q.lower().endswith(suff.lower()) else (q.rstrip() + " " + suff)

def resolve_image(img_root: Path, image_name: str) -> Path:
    """Resolves the image path robustly."""
    p = Path(image_name)

    # 1) Use absolute path directly if it exists
    if p.is_absolute() and p.exists():
        return p

    # 2) Direct concatenation
    cand = img_root / image_name
    if cand.exists():
        return cand

    # 3) Try different extensions based on stem
    stem = Path(image_name).stem
    for ext in VALID_IMG_EXT:
        c = img_root / f"{stem}{ext}"
        if c.exists():
            return c

    # 4) Global fallback search (slow but comprehensive)
    hits = [h for h in img_root.rglob(f"{stem}*") if h.is_file()]
    if hits:
        return hits[0]

    raise FileNotFoundError(f"Image not found for {image_name} under {img_root}")

def build_pope_ann(records, img_root: Path, variant: str = "random"):
    """
    Builds the standard POPE format annotation.
    Input format common example:
    {
      "question_id": 0,
      "image": "COCO_val2014_000000000042.jpg",
      "text": "Is there a fork in the image?",
      "label": "no"
    }
    """
    items = []
    skipped_no_img = 0
    skipped_no_q = 0
    skipped_bad_label = 0
    missing_img = 0

    for idx, rec in enumerate(records):
        # Field compatibility checks
        qid = rec.get("question_id", idx)
        img = rec.get("image") or rec.get("image_path") or rec.get("image_id")
        question = rec.get("text") or rec.get("question") or rec.get("prompt")
        label_raw = rec.get("label") or rec.get("answer") or rec.get("gt")
        label = _norm_yesno(label_raw)

        if img is None:
            skipped_no_img += 1
            continue
        if question is None:
            skipped_no_q += 1
            continue
        if label not in ("yes", "no"):
            skipped_bad_label += 1
            continue

        try:
            img_path = resolve_image(img_root, img)
        except FileNotFoundError as e:
            print(f"[warn] {e}")
            missing_img += 1
            # Keeping the item allows tracking missing files, evaluation step needs to handle it.
            img_path = (img_root / img).resolve()

        item = {
            "question_id": str(qid),
            "image_path": str(img_path),
            # Include both question and prompt fields for maximum compatibility
            "question": ensure_suffix(question),
            "prompt": ensure_suffix(question),
            # Include both label and answers to be compatible with different evalkit implementations
            "label": label,
            "answers": [label],
            "variant": variant  # random/popular/adversarial
        }
        items.append(item)

    total = len(records)
    built = len(items)
    print(f"[build_ann] total={total}, built={built}, "
          f"skip_no_img={skipped_no_img}, skip_no_q={skipped_no_q}, skip_bad_label={skipped_bad_label}, "
          f"missing_img_files={missing_img}")
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

    # POPE annotation files are often named: coco_pope_random.json / coco_pope_popular.json / coco_pope_adversarial.json
    ann_file = pope_root / f"coco_pope_{args.variant}.json"
    if not ann_file.exists():
        # Try other common names
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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[build_ann] Saved to {out_path}")

if __name__ == "__main__":
    main()