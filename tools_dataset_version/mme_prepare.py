#!/usr/bin/env python3
# tools/mme_prepare.py
"""检查和准备 MME 数据集"""

import json
from pathlib import Path
import argparse

def check_mme_structure(mme_root: Path):
    """检查 MME 数据集的结构"""
    print(f"\n{'='*60}")
    print(f"Checking MME structure: {mme_root}")
    print(f"{'='*60}\n")
    
    if not mme_root.exists():
        print(f"[error] Directory not found: {mme_root}")
        return
    
    # 列出所有子任务
    subtasks = [d for d in mme_root.iterdir() if d.is_dir()]
    print(f"Found {len(subtasks)} subtasks:\n")
    
    for subtask_dir in sorted(subtasks):
        subtask = subtask_dir.name
        print(f"📁 {subtask}/")
        
        # 检查是否有 questions_answers_YN 目录
        qa_yn_dir = subtask_dir / "questions_answers_YN"
        if qa_yn_dir.exists():
            txt_files = list(qa_yn_dir.glob("*.txt"))
            print(f"   ✓ questions_answers_YN/ ({len(txt_files)} txt files)")
        
        # 检查是否有图片
        img_files = []
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            img_files.extend(list(subtask_dir.rglob(f"*{ext}")))
        if img_files:
            print(f"   ✓ {len(img_files)} images")
        
        # 检查是否有 JSON 标注
        json_files = list(subtask_dir.glob("*.json"))
        if json_files:
            print(f"   ✓ {len(json_files)} JSON files: {[f.name for f in json_files]}")
        
        print()

def main():
    ap = argparse.ArgumentParser(description="Check MME dataset structure")
    ap.add_argument("--mme_root", required=True,
                    help="MME dataset root directory")
    args = ap.parse_args()
    
    mme_root = Path(args.mme_root).resolve()
    check_mme_structure(mme_root)

if __name__ == "__main__":
    main()