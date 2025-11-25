#!/usr/bin/env python3
# scripts/dataset_download.py
"""
File-layer dataset downloader using huggingface_hub Python API (no CLI subcommands).
- Lists files via HfApi (list_repo_files / list_repo_tree)
- Downloads matching files via hf_hub_download
- Auto-extracts .zip

Prerequisites:
  pip install -U huggingface_hub
  huggingface-cli login   # or: from huggingface_hub import login
"""

import argparse
import fnmatch
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ===================== Dataset Configs =====================
DATASET_CONFIGS = {
    "coco_val": {
        "is_special": "coco",
        "split": "val2014",
        "description": "COCO 2014 Validation Images",
        "size": "~8 GB",
        "note": "Downloaded directly via URL (wget/curl/urllib).",
        "out_subdir": "coco", # Optional output subdirectory name
    },
    "coco_train": {
        "is_special": "coco",
        "split": "train2014",
        "description": "COCO 2014 Training Images",
        "size": "~13 GB",
        "note": "Downloaded directly via URL (wget/curl/urllib).",
        "out_subdir": "coco", # Optional output subdirectory name
    },
    "pope": {
        "hf_repo": "lmms-lab/POPE",
        "repo_type": "dataset",
        "description": "POPE (Polling-based Object Probing Evaluation)",
        "size": "~510 MB",
        # HF only has parquet files; the actual JSONs are on GitHub:
        "includes": [  # Keep these for potential future updates, but they might not match current HF files
            "coco_pope_random.json",
            "coco_pope_popular.json",
            "coco_pope_adversarial.json",
        ],
        "github_raw": [
            "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json",
            "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_popular.json",
            "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json",
        ],
    },
    "mme": {
        # Note: Current use is darkyarding/MME. Replace with lmms-lab/MME if needed.
        "hf_repo": "darkyarding/MME",
        "repo_type": "dataset",
        "description": "MME (Multi-Modal Evaluation) benchmark",
        "size": "~3 GB",
        "note": "14 subtask images and annotations; distributed as a zip file.",
        "includes": [
            "MME_Benchmark_release_version.zip",
            "*MME*Benchmark*release*version*.zip",
            "*.json", "*.txt",
        ],
    },
    "mmbench": {
        "hf_repo": "lmms-lab/MMBench",
        "repo_type": "dataset",
        "description": "MMBench comprehensive evaluation",
        "size": "~2 GB",
        "includes": ["*MMBench*Images*.zip", "*image*zip", "*.tsv", "*.json", "*.txt"],
    },
    "seed_bench": {
        "hf_repo": "lmms-lab/SEED-Bench",
        "repo_type": "dataset",
        "description": "SEED-Bench",
        "size": "~5 GB",
        "includes": ["*SEED*Images*.zip", "*.tsv", "*.json", "*.txt"],
    },
    "vqav2": {
        "hf_repo": "HuggingFaceM4/VQAv2",
        "repo_type": "dataset",
        "description": "VQA v2.0",
        "size": "~100 MB (annotations only)",
        "note": "Images need COCO separately",
        "includes": ["annotations/*", "v2_mscoco_*", "*.json", "*.txt"],
    },
    "llava_bench": {
        "hf_repo": "lmms-lab/llava-bench-in-the-wild",
        "repo_type": "dataset",
        "description": "LLaVA-Bench-in-the-Wild",
        "size": "~500 MB",
        "includes": ["*zip", "*.json", "*.tsv", "*.txt"],
    },
    "coco_caption": {
        "is_special": "coco_caption",
        "description": "COCO 2014 caption annotations (captions_train2014/val2014.json)",
        "size": "~100 MB",
        "note": "Use together with coco_train/coco_val images.",
        "out_subdir": "coco",
    },
}
# ===========================================================

def check_hf_cli():
    """Checks if the huggingface-cli command is available in the system path."""
    if not shutil.which("huggingface-cli"):
        print("❌ Error: huggingface-cli not found!")
        print("  pip install huggingface_hub")
        return False
    return True

def check_hf_auth():
    """Checks the HuggingFace login status (does not force exit)."""
    r = subprocess.run(["huggingface-cli", "whoami"], capture_output=True, text=True)
    if r.returncode != 0:
        print("⚠️  Not logged in to HuggingFace. If the repo is gated, run: huggingface-cli login")
        return False
    print("✓ Logged in as:", r.stdout.strip().splitlines()[-1])
    return True

# -------------------- API-based list & download --------------------
def list_repo_files_api(repo_id: str, repo_type: str) -> list[str]:
    """Lists all files in a HuggingFace repository using the HfApi."""
    from huggingface_hub import HfApi
    api = HfApi()
    # Prefer list_repo_files; fallback to list_repo_tree for older versions
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        return list(files)
    except Exception:
        # Fallback: list_repo_tree
        tree = api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, recursive=True)
        return [e.path for e in tree]

def hf_download_to_dir(repo_id: str, repo_type: str, filename: str, out_dir: Path) -> Path:
    """
    Downloads a single file to out_dir (ensuring it's not just in the cache):
    - Prefers hf_hub_download(local_dir=..., local_dir_use_symlinks=False)
    - Fallbacks to copy if local_dir_use_symlinks is not available in older versions
    """
    from huggingface_hub import hf_hub_download
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # New version: Downloads directly into out_dir
        p = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
        return Path(p)
    except TypeError:
        # Older version: Only returns cache path; we copy it to out_dir
        p_cache = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename)
        dst = out_dir / Path(filename).name
        if Path(p_cache) != dst:
            shutil.copy2(p_cache, dst)
        return dst

def download_urls(urls: list[str], out_dir: Path) -> list[Path]:
    """Downloads files from a list of raw URLs (e.g., GitHub raw) to the output directory."""
    import urllib.request
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for url in urls:
        dst = out_dir / Path(url).name
        try:
            if shutil.which("wget"):
                subprocess.run(["wget", "-O", str(dst), url], check=True)
            elif shutil.which("curl"):
                subprocess.run(["curl", "-L", "-o", str(dst), url], check=True)
            else:
                urllib.request.urlretrieve(url, str(dst))
            out_paths.append(dst)
        except Exception as e:
            print(f"  ❌ Download failed: {url} -> {e}")
    return out_paths

def download_files_via_api(repo_id: str, repo_type: str, includes: list[str], out_dir: Path) -> list[Path]:
    """Lists files in HF repo and downloads those matching the include patterns."""
    all_files = list_repo_files_api(repo_id, repo_type)
    pats = includes or ["*"]
    # Find files matching any of the patterns
    matched = sorted({f for f in all_files if any(fnmatch.fnmatch(f, pat) for pat in pats)})
    if not matched:
        print(f"⚠️  No files matched: {pats}")
        return []
    print(f"[list] Matched {len(matched)} files:")
    for f in matched[:20]:
        print("  -", f)
    if len(matched) > 20:
        print(f"  ... (+{len(matched)-20} more)")
    out_paths = []
    for rel in matched:
        try:
            out_paths.append(hf_download_to_dir(repo_id, repo_type, rel, out_dir))
        except Exception as e:
            print(f"  ❌ Download failed: {rel} -> {e}")
    return out_paths

# -------------------- Utils --------------------
def extract_all_zips(root: Path, keep_zip: bool = False):
    """Recursively finds and extracts all zip files in the given root directory."""
    zips = list(root.rglob("*.zip"))
    if not zips:
        print("ℹ️  No zip files found to extract.")
        return
    print(f"[extract] Found {len(zips)} zip files. Extracting...")
    for z in zips:
        try:
            print(f"  - {z}")
            with zipfile.ZipFile(z, "r") as zip_ref:
                # Extract to the directory where the zip file resides
                zip_ref.extractall(z.parent)
            if not keep_zip:
                z.unlink()
        except Exception as e:
            print(f"  ⚠️  Failed to extract {z}: {e}")

def list_datasets():
    """Prints a formatted list of all available dataset configurations."""
    print("\n" + "="*80)
    print("Available Datasets (file-layer via API)")
    print("="*80 + "\n")
    for name, cfg in DATASET_CONFIGS.items():
        print(f"📦 {name}")
        if 'hf_repo' in cfg:
            print(f"   Repo: {cfg['hf_repo']} ({cfg.get('repo_type','dataset')})")
        print(f"   Desc: {cfg['description']}")
        if 'size' in cfg: print(f"   Size: {cfg['size']}")
        if 'note' in cfg: print(f"   Note: {cfg['note']}")
        inc = cfg.get("includes") or []
        if inc:
            print(f"   Includes: {', '.join(inc[:3])}{' ...' if len(inc)>3 else ''}")
        print()

def download_coco_images(output_dir: Path, split: str = "val2014"):
    """Downloads and extracts COCO images (val2014 or train2014) using external tools or fallbacks."""
    import zipfile
    urls = {
        "val2014": "http://images.cocodataset.org/zips/val2014.zip",
        "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    }
    if split not in urls:
        print(f"❌ Unknown COCO split: {split}")
        return False
    url = urls[split]
    zip_path = output_dir / f"{split}.zip"
    final_dir = output_dir / split
    if final_dir.exists() and any(final_dir.iterdir()):
        print(f"✓ {split} already exists at {final_dir}")
        return True
    print(f"[download] COCO {split} -> {zip_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Use system tools for large file download reliability
        if shutil.which("wget"):
            subprocess.run(["wget", "-O", str(zip_path), url], check=True)
        elif shutil.which("curl"):
            subprocess.run(["curl", "-L", "-o", str(zip_path), url], check=True)
        else:
            print("❌ Neither wget nor curl found.")
            return False
        
        print(f"[extract] {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        zip_path.unlink(missing_ok=True)
        print(f"✓ COCO {split} extracted to: {final_dir}")
        return True
    except Exception as e:
        print("❌ Failed:", e)
        zip_path.unlink(missing_ok=True)
        return False

def download_coco_captions(output_dir: Path) -> bool:
    """
    Downloads and extracts COCO 2014 caption annotations:
    - captions_train2014.json
    - captions_val2014.json

    The official zip is annotations_trainval2014.zip, which contains
    an inner 'annotations/' directory.
    """
    import zipfile

    url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    zip_path = output_dir / "annotations_trainval2014.zip"

    inner_dir = output_dir / "annotations"
    train_json = inner_dir / "captions_train2014.json"
    val_json = inner_dir / "captions_val2014.json"

    if train_json.exists() and val_json.exists():
        print(f"✓ COCO caption annotations already exist at {inner_dir}")
        return True

    print(f"[download] COCO caption annotations -> {zip_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if shutil.which("wget"):
            subprocess.run(["wget", "-O", str(zip_path), url], check=True)
        elif shutil.which("curl"):
            subprocess.run(["curl", "-L", "-o", str(zip_path), url], check=True)
        else:
            import urllib.request
            urllib.request.urlretrieve(url, str(zip_path))

        print(f"[extract] {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        zip_path.unlink(missing_ok=True)

        if train_json.exists() and val_json.exists():
            print(f"✓ COCO captions extracted to: {inner_dir}")
            return True
        else:
            print(f"❌ captions_train2014.json / captions_val2014.json not found under {inner_dir}")
            return False
    except Exception as e:
        print("❌ Failed to download COCO captions:", e)
        zip_path.unlink(missing_ok=True)
        return False

# -------------------- Main CLI --------------------
def main():
    parser = argparse.ArgumentParser(description="File-layer dataset downloader (HF API)")
    dataset_choices = list(DATASET_CONFIGS.keys())
    parser.add_argument("--dataset", nargs="+", choices=dataset_choices)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./datasets")
    parser.add_argument("--skip-auth-check", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return
    if not args.dataset: # check --dataset
        parser.print_help()
        print("\n❌ Error: Please specify --dataset or --list")
        return

    if not check_hf_cli():
        sys.exit(1)
    if not args.skip_auth_check:
        check_hf_auth()

    output_root = Path(args.output_dir).resolve()
    print(f"\n📁 Output directory: {output_root}\n")

    success = 0
    failed = []

    # HF/COCO Datasets Download
    if args.dataset:
        for name in args.dataset:
            cfg = DATASET_CONFIGS[name]
            
            # configure output directory
            out_subdir = cfg.get("out_subdir", name)
            out_dir = output_root / out_subdir
            
            print("\n" + "="*80)
            print(f"Dataset: {name}")
            if 'hf_repo' in cfg:
                print(f"Repo: {cfg['hf_repo']}  (type={cfg.get('repo_type','dataset')})")
            print(f"Output: {out_dir}")
            print("="*80 + "\n")

            try:
                # coco special handling
                if cfg.get("is_special") == "coco":
                    if download_coco_images(output_root / out_subdir, cfg["split"]):
                        print(f"✅ Done: {name}")
                        success += 1
                    else:
                        raise RuntimeError("COCO download failed")
                    continue # COCO download handled, skip to next
                elif cfg.get("is_special") == "coco_caption":
                    if download_coco_captions(out_dir):
                        print(f"✅ Done: {name}")
                        success += 1
                    else:
                        raise RuntimeError("COCO caption download failed")
                    continue  # COCO caption handled

                # 1. Download files from HuggingFace API
                files = download_files_via_api(
                    repo_id=cfg["hf_repo"],
                    repo_type=cfg.get("repo_type", "dataset"),
                    includes=cfg.get("includes", []),
                    out_dir=out_dir,
                )

                # 2. GitHub fallback (triggered only if no files were matched on HF)
                if (not files) and cfg.get("github_raw"):
                    print("ℹ️  No annotation files matched on HF; falling back to GitHub raw links ...")
                    files = download_urls(cfg["github_raw"], out_dir)

                if not files:
                    raise RuntimeError("no files matched or downloaded")

                # 3. Extraction
                if not args.no_extract:
                    extract_all_zips(out_dir, keep_zip=args.keep_zip)
                
                print(f"✅ Done: {name}")
                success += 1
            except Exception as e:
                print(f"❌ Failed {name}: {e}")
                failed.append(name)

    # Final Summary
    total = len(args.dataset or [])
    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)
    print(f"✅ Successfully processed: {success}/{total}")
    if failed:
        print("❌ Failed:", ", ".join(failed))
    print(f"📁 Output directory: {output_root}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()