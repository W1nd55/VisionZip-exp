#!/usr/bin/env python3
# scripts/evalkit.py
from pathlib import Path
import argparse
import random
import torch

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
visionzip_parent_dir = os.path.join(current_dir, os.pardir, 'models/VisionZip')
if visionzip_parent_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(visionzip_parent_dir))
# Ensure local LLaVA package is importable when running from the repo root.
llava_parent_dir = os.path.join(current_dir, os.pardir, 'models/LLaVA')
if llava_parent_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(llava_parent_dir))

# # Also make SparseVLMs importable (some runs reference that package directly)
# sparsevlm_parent_dir = os.path.join(current_dir, os.pardir, 'models/SparseVLMs')
# if sparsevlm_parent_dir not in sys.path:
#     sys.path.insert(0, os.path.abspath(sparsevlm_parent_dir))



from scripts.evaluate import Evaluator
from scripts.metric import DelayStats

# -------------------- YAML utils --------------------
def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML at {path} must be a mapping, got {type(obj)}")
    return obj

def _get(d: dict, key: str, default=None):
    # Safely get a value from a dictionary, returning default if not a dict or key is missing
    return d.get(key, default) if isinstance(d, dict) else default

# -------------------- Model builders --------------------
def build_model_llava(model_cfg: dict):
    from scripts.model import LlavaModel
    kwargs = {
        "model_path":     _get(model_cfg, "model_path"),
        "temperature":    float(_get(model_cfg, "temperature", 0.0)),
        "max_new_tokens": int(_get(model_cfg, "max_new_tokens", 16)),
    }
    return LlavaModel(**kwargs)

def build_model_llava_vzip(model_cfg: dict):
    from scripts.model import LlavaVisionZipModel
    kwargs = {
        "model_path":     _get(model_cfg, "model_path"),
        "dominant":       int(_get(model_cfg, "dominant", 54)),
        "contextual":     int(_get(model_cfg, "contextual", 10)),
        "temperature":    float(_get(model_cfg, "temperature", 0.0)),
        "max_new_tokens": int(_get(model_cfg, "max_new_tokens", 16)),
    }
    return LlavaVisionZipModel(**kwargs)

def build_model_llava_vzip_hybrid_attn(model_cfg: dict):
    from scripts.model import LlavaVisionZipModelHybridAttn
    alpha_list = _get(model_cfg, "vision_zip_alpha", [1.2, 0.9, 0.2])
    alpha_config = tuple(float(x) for x in alpha_list)
    
    kwargs = {
        "model_path":     _get(model_cfg, "model_path"),
        "dominant":       int(_get(model_cfg, "dominant", 54)),
        "contextual":     int(_get(model_cfg, "contextual", 10)),
        "temperature":    float(_get(model_cfg, "temperature", 0.0)),
        "max_new_tokens": int(_get(model_cfg, "max_new_tokens", 16)),
        "alpha_config": alpha_config,
    }
    return LlavaVisionZipModelHybridAttn(**kwargs)

def build_model_llava_vzip_dynamic_k(model_cfg: dict):
    from scripts.model import LlavaVisionZipModelDynamicK
    
    kwargs = {
        "model_path":     _get(model_cfg, "model_path"),
        "dominant":       int(_get(model_cfg, "dominant", 54)),
        "contextual":     int(_get(model_cfg, "contextual", 10)),
        "temperature":    float(_get(model_cfg, "temperature", 0.0)),
        "max_new_tokens": int(_get(model_cfg, "max_new_tokens", 16)),
    }
    return LlavaVisionZipModelDynamicK(**kwargs)

def build_model_sparsezip(model_cfg: dict):
    """Builds the SparseZip-enabled LLaVA model as a distinct model_type.

    This is functionally similar to build_model_llava_vzip but provides a clear
    entrypoint and selection via model_type: 'sparsezip'.
    """
    from scripts.model import LlavaSparseZipModel
    alpha_list = _get(model_cfg, "vision_zip_alpha", [1.0, 0, 0])
    alpha_config = tuple(float(x) for x in alpha_list)
    kwargs = {
        "model_path":     _get(model_cfg, "model_path"),
        "dominant":       int(_get(model_cfg, "dominant", 54)),
        "contextual":     int(_get(model_cfg, "contextual", 10)),
        "temperature":    float(_get(model_cfg, "temperature", 0.0)),
        "max_new_tokens": int(_get(model_cfg, "max_new_tokens", 16)),
        "sparsezip_cfg":  _get(model_cfg, "sparsezip", {}),
        "alpha_config": alpha_config,
    }
    return LlavaSparseZipModel(**kwargs)

def build_model_sparsevlm(model_cfg: dict):
    from scripts.model import SparseVLMModel
    kwargs = {
        "model_path":      _get(model_cfg, "model_path"),
        "temperature":     float(_get(model_cfg, "temperature", 0.0)),
        "max_new_tokens":  int(_get(model_cfg, "max_new_tokens", 128)),
        "retained_tokens": int(_get(model_cfg, "retained_tokens", 192)),
        "conv_mode":       _get(model_cfg, "conv_mode", "llava_v1"),
    }
    return SparseVLMModel(**kwargs)

def build_model_hybrid(model_cfg: dict):
    from scripts.model import LlavaVisionZipHybridModel
    kwargs = {
        "model_path":     _get(model_cfg, "model_path"),
        "dominant":       int(_get(model_cfg, "dominant", 54)),  
        "contextual":     int(_get(model_cfg, "contextual", 10)), 
        "temperature":    float(_get(model_cfg, "temperature", 0.0)),
        "max_new_tokens": int(_get(model_cfg, "max_new_tokens", 16)),
    }
    if kwargs["model_path"] is None:
        raise ValueError("model.model_path is required for hybrid model")
    return LlavaVisionZipHybridModel(**kwargs)

def build_model_textaware(model_cfg: dict):
    from scripts.model import LlavaVisionZipTextAwareModel
    kwargs = {
        "model_path":     _get(model_cfg, "model_path"),
        "dominant":       int(_get(model_cfg, "dominant", 54)),  
        "contextual":     int(_get(model_cfg, "contextual", 10)), 
        "temperature":    float(_get(model_cfg, "temperature", 0.0)),
        "max_new_tokens": int(_get(model_cfg, "max_new_tokens", 16)),
    }
    if kwargs["model_path"] is None:
        raise ValueError("model.model_path is required for hybrid model")
    return LlavaVisionZipTextAwareModel(**kwargs)

def build_model(model_cfg: dict):
    mtype = (_get(model_cfg, "model_type", "llava") or "llava").lower()
    if mtype == "llava":
        return build_model_llava(model_cfg)
    if mtype == "llava_vzip":
        return build_model_llava_vzip(model_cfg)
    if mtype == "llava_vzip_hybridattn":
        return build_model_llava_vzip_hybrid_attn(model_cfg)
    if mtype == "llava_vzip_dynamick":
        return build_model_llava_vzip_dynamic_k(model_cfg)
    if mtype == "sparsezip":
        return build_model_sparsezip(model_cfg)
    if mtype == "sparsevlm":
        return build_model_sparsevlm(model_cfg)
    if mtype == "hybrid":
        return build_model_hybrid(model_cfg)
    if mtype == "textaware":
        return build_model_textaware(model_cfg)
    raise ValueError(f"Unknown model_type: {mtype}")

# -------------------- Dataset/Metric builders --------------------
def build_dataset_and_metrics(dataset_name: str, ann_path: str, limit: int | None):
    # Build the corresponding dataset object and list of metrics based on the dataset name
    if dataset_name == "vqa":
        from scripts.dataset import VQAv2Dataset
        from scripts.metric import ExactMatch, VQASoftAcc
        dataset = VQAv2Dataset(ann_path, limit=limit)
        metrics = [ExactMatch(), VQASoftAcc(), DelayStats()]
    elif dataset_name == "mme":
        from scripts.dataset import MMEDataset
        from scripts.metric import MMEAcc, MMEAccPlus
        dataset = MMEDataset(ann_path, limit=limit)
        metrics = [MMEAcc(), MMEAccPlus(), DelayStats()]
    elif dataset_name == "pope":
        from scripts.dataset import POPEDataset
        from scripts.metric import POPEAcc, POPEPrecisionRecallF1
        dataset = POPEDataset(ann_path, limit=limit)
        metrics = [POPEAcc(), POPEPrecisionRecallF1(), DelayStats()]
    elif dataset_name == "coco_caption":
        # COCO Caption: ann_path -> captions_*.json
        from scripts.dataset import COCOCaptionDataset
        from scripts.metric import (
            CaptionBLEU,
            CaptionROUGEL,
            CaptionMETEOR,
            CaptionCIDEr,
        )
        dataset = COCOCaptionDataset(ann_path, limit=limit)
        metrics = [
            CaptionBLEU(),
            CaptionROUGEL(),
            CaptionMETEOR(),
            CaptionCIDEr(),
            DelayStats(),
        ]
    elif dataset_name == "docvqa":
        # DocVQA: Single-Page Document VQA (Task 1)
        from scripts.dataset import DocVQADataset
        from scripts.metric import DocVQAANLS, ExactMatch

        dataset = DocVQADataset(
            ann_path=ann_path,
            limit=limit,
            image_root=None,
        )
        metrics = [
            DocVQAANLS(),
            ExactMatch(),
            DelayStats(),
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset, metrics

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # YAML (Required)
    parser.add_argument("--cfg", type=str, required=True, help="YAML configuration file, e.g., config/visionzip.yaml")

    # Arguments only passed to override YAML values (None = no override)
    # Override 'model' section
    parser.add_argument("--model_type", type=str, default=None, choices=["llava_vzip", "sparsezip", "sparsevlm", "llava", "hybrid", "textaware"]) 
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dominant", type=int, default=None)
    parser.add_argument("--contextual", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--retained_tokens", type=int, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)

    # Override 'runner' section
    parser.add_argument("--dataset", type=str, default=None, choices=["vqa","mme","pope","coco_caption", "docvqa"])
    parser.add_argument("--ann_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    # 1) Load YAML configuration
    cfg = _load_yaml(Path(args.cfg))
    model_cfg  = (_get(cfg, "model",  {}) or {}).copy()
    runner_cfg = (_get(cfg, "runner", {}) or {}).copy()

    # 2) CLI Overrides (Only for non-None values)
    # Override model configuration
    for k in ["model_type","model_path","dominant","contextual","temperature","max_new_tokens",
              "retained_tokens","conv_mode"]:
        v = getattr(args, k, None)
        if v is not None:
            model_cfg[k] = v

    # Override runner configuration
    for k in ["dataset","ann_path","output_dir","warmup","seed","limit"]:
        v = getattr(args, k, None)
        if v is not None:
            runner_cfg[k] = v

    # 3) Build Model
    model = build_model(model_cfg)

    # 4) Final Runner Settings
    dataset    = _get(runner_cfg, "dataset", "vqa")
    ann_path   = _get(runner_cfg, "ann_path")
    output_dir = _get(runner_cfg, "output_dir", "./outputs_eval")
    warmup     = int(_get(runner_cfg, "warmup", 3))
    seed       = int(_get(runner_cfg, "seed", 42))
    limit      = _get(runner_cfg, "limit", None)
    # Convert limit to integer if it was read as a digit string
    if isinstance(limit, str) and limit.isdigit():
        limit = int(limit)
    if not ann_path:
        # Annotation path is mandatory
        raise ValueError("runner.ann_path is required (or pass --ann_path to override)")

    # 5) Set Seeds
    random.seed(seed)
    torch.manual_seed(seed)

    # 6) Build Dataset and Metrics
    dataset_obj, metrics = build_dataset_and_metrics(dataset, ann_path, limit)

    # 7) Run Evaluation
    evaluator = Evaluator(model, dataset_obj, metrics, output_dir=output_dir, warmup=warmup, seed=seed)
    evaluator.run(limit=limit)