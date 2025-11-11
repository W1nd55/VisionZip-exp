# tools/mme_build_ann.py
import os, json, argparse, glob, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def find_annotation_file(subdir: Path) -> Path | None:
    """
    Finds the primary annotation file within a subtask directory.
    Prioritizes common filenames, then falls back to searching by extension.
    """
    # Priority for common file names
    candidates = [
        "qa.json", "questions.json", "annotation.json",
        "mme_qa.json", "pairs.json", "eval.json",
    ]
    for c in candidates:
        p = subdir / c
        if p.exists():
            return p

    # Fallback: Scan for common annotation file extensions
    js = list(subdir.glob("*.json"))
    if js: return js[0]
    jl = list(subdir.glob("*.jsonl"))
    if jl: return jl[0]
    cs = list(subdir.glob("*.csv"))
    if cs: return cs[0]
    ts = list(subdir.glob("*.tsv"))
    if ts: return ts[0]
    return None

def load_annotation(path: Path) -> List[Dict[str, Any]]:
    """Loads annotation records from JSON, JSONL, CSV, or TSV files."""
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

def guess_fields(rec: dict) -> Tuple[str | None, str | None, str | None, str | None, str | None]:
    """
    Attempts to guess fields based on common naming conventions.
    Returns: image, q_pos, a_pos, q_neg, a_neg
    (Allows records to be single or contain the pos/neg pair.)
    """
    # Candidates for common field names
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

    # If it's a single yes/no record, it needs aggregation in the upper layer; return as is
    return image, q_pos, a_pos, q_neg, a_neg

def build_pairs(records: List[Dict[str, Any]], img_root: Path) -> List[Dict[str, Any]]:
    """
    Attempts to assemble the annotation into the MME "two questions per image (pos/neg)" structure.
    Supports two sources:
      1) Records already contain q_pos/q_neg fields.
      2) Records are single yes/no questions, needing merging by image_id into a pair.
    """
    paired = []
    singles_by_image = {}

    for rec in records:
        image, q_pos, a_pos, q_neg, a_neg = guess_fields(rec)

        # Fallback: some files use 'image_id' for the image name
        if image is None and "image_id" in rec:
            image = rec["image_id"]

        # Fallback: only one question/answer field is present
        if q_pos is None and q_neg is None:
            q = rec.get("question") or rec.get("q") or rec.get("prompt")
            a = (rec.get("answer") or rec.get("a") or rec.get("label") or "").strip().lower()
            if image is None:
                # Image field still missing after all attempts
                raise ValueError(f"Cannot find image field in record: {rec}")
            singles_by_image.setdefault(image, []).append((q, a))
            continue

        if image is None:
            raise ValueError(f"Missing image field in paired record: {rec}")

        # Standardize yes/no answers
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

    # Process singles -> pairs (Assuming one 'yes' question and one 'no' question per image)
    for image, qa_list in singles_by_image.items():
        # Select one yes question and one no question (take the first two if multiple exist)
        yes_q = next((q for q,a in qa_list if a in ("yes","y","1","true")), None)
        no_q  = next((q for q,a in qa_list if a in ("no","n","0","false")), None)
        if yes_q is None or no_q is None:
            # Incomplete data, choose to skip or raise error; here we skip
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

def ensure_suffix(q: str | None, suffix: str) -> str | None:
    """Appends the required suffix to the question if it's missing."""
    if q is None:
        return None
    q = q.strip()
    suff = suffix.strip()
    return q if q.lower().endswith(suff.lower()) else (q.rstrip() + " " + suff)

def resolve_image(img_root: Path, image_name: str) -> Path:
    """
    Attempts to find the full path to the image file under img_root.
    Handles absolute paths, direct match, stem + extension match, and rglob fallback.
    """
    p = Path(image_name)
    if p.is_absolute() and p.exists():
        return p
    # Try direct path concatenation
    cand = img_root / image_name
    if cand.exists():
        return cand
    # Broadcast match (try common extensions)
    stem = Path(image_name).stem
    for ext in VALID_IMG_EXT:
        c = img_root / f"{stem}{ext}"
        if c.exists():
            return c
    # Final full search (slow, but robust)
    hits = list(img_root.rglob(f"{stem}.*"))
    if hits:
        return hits[0]
    # Check if the file is directly in the subtask root
    if Path(image_name).suffix not in VALID_IMG_EXT:
        for ext in VALID_IMG_EXT:
            c = img_root / f"{image_name}{ext}"
            if c.exists():
                return c

    raise FileNotFoundError(f"Image not found for {image_name} under {img_root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mme_root", required=True, help="Root directory of MME_Benchmark (containing subtask dirs like existence/ color/ ...)")
    ap.add_argument("--subtask", required=True, help="Subtask directory name, e.g., existence / color / OCR ...")
    ap.add_argument("--out", required=True, help="Output JSON path")
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

    # Ensure output directory exists before saving
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(paired, f, ensure_ascii=False, indent=2)
    print(f"[build_ann] Saved to {args.out}")

if __name__ == "__main__":
    main()