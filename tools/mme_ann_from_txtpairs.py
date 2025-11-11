#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, re
from pathlib import Path

# Case-insensitive compatible image extensions
VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
                 ".JPG", ".JPEG", ".PNG", ".WEBP", ".BMP"}

def ensure_suffix(q: str) -> str:
    q = (q or "").strip()
    suff = "Please answer yes or no."
    return q if q.lower().endswith(suff.lower()) else (q.rstrip() + " " + suff)

def parse_line(line: str):
    """
    Input a line formatted like:
      "Is this ... ? Please answer yes or no.   Yes"
    Returns: (question, 'yes'/'no')
    """
    s = (line or "").strip()
    if not s:
        return None, None
    # Search for the trailing Yes/No label
    m = re.search(r"\b(yes|no)\b\s*$", s, flags=re.IGNORECASE)
    if not m:
        return s, None
    label = m.group(1).lower()
    q = s[:m.start()].rstrip(" \t:;,.")
    return q.strip(), label

def find_image(img_root: Path, stem: str) -> Path | None:
    # First, try direct concatenation in the same directory
    for ext in VALID_IMG_EXT:
        p = img_root / f"{stem}{ext}"
        if p.exists():
            return p
    # Recursive search (slower)
    hits = []
    for ext in VALID_IMG_EXT:
        hits.extend(img_root.rglob(f"{stem}{ext}"))
    if hits:
        # Choose the one with the shortest path
        hits.sort(key=lambda x: (len(str(x).split(os.sep)), str(x)))
        return hits[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask_dir", required=True, help="Root directory of the subtask, e.g., .../MME_Benchmark/OCR")
    ap.add_argument("--qa_dirname", default="questions_answers_YN", help="Default subdirectory name containing two-line txt files")
    ap.add_argument("--qa_glob", default=None, help="Recursive matching pattern, e.g., '**/*.txt' or 'questions_answers_YN/*.txt'")
    ap.add_argument("--out", required=True, help="Output annotation JSON path")
    args = ap.parse_args()

    subtask_dir = Path(args.subtask_dir).resolve()

    # Collect txt files: prioritize qa_glob; otherwise use fixed directory
    if args.qa_glob:
        txt_files = sorted(subtask_dir.rglob(args.qa_glob))
        pattern_desc = args.qa_glob
    else:
        qa_dir = subtask_dir / args.qa_dirname
        if not qa_dir.exists():
            # Error message translated to English
            raise FileNotFoundError(f"{qa_dir} does not exist; consider using --qa_glob '**/*.txt' instead")
        txt_files = sorted(qa_dir.glob("*.txt"))
        pattern_desc = f"{args.qa_dirname}/*.txt"

    print(f"[build] Found {len(txt_files)} txt files under {subtask_dir} (pattern={pattern_desc})")

    items = []              # ←←← CRITICAL: Initialization is here to prevent NameError
    miss_img = 0
    bad_txt = 0

    for tf in txt_files:
        stem = tf.stem
        # Read non-empty lines
        lines = [ln.strip() for ln in tf.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        if len(lines) < 2:
            bad_txt += 1
            continue

        q1, a1 = parse_line(lines[0])
        q2, a2 = parse_line(lines[1])

        # Error tolerance: if no label is parsed, default to 1st=yes, 2nd=no
        if a1 not in ("yes", "no"): a1 = "yes"
        if a2 not in ("yes", "no"): a2 = "no"

        # Normalize to pos/neg (pos=Yes, neg=No)
        if a1 == "yes" and a2 == "no":
            q_pos, q_neg = q1, q2
        elif a1 == "no" and a2 == "yes":
            q_pos, q_neg = q2, q1
        else:
            bad_txt += 1
            continue

        img_path = find_image(subtask_dir, stem)
        if img_path is None:
            miss_img += 1
            continue

        items.append({
            "image_id": stem,
            "image_path": str(img_path),
            "q_pos": ensure_suffix(q_pos),
            "a_pos": "yes",
            "q_neg": ensure_suffix(q_neg),
            "a_neg": "no"
        })

    print(f"[build] Built {len(items)} items, missed_img={miss_img}, bad_txt={bad_txt}")
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build] Saved to {outp}")

if __name__ == "__main__":
    main()