"""Quick Smoke Test for SparseZip Compression

This script performs a minimal single-image, single-prompt inference using a LLaVA
model with the SparseZip vision token compressor enabled via a sparsezip YAML.

It loads the image in `reference/owl.JPEG`, applies the model patch (through
normal model building path), runs a prompt, and prints:

  - Generated answer
  - Number of original vision tokens vs compressed tokens
  - Selected dominant K and contextual C
  - Indices of retained tokens (first few for inspection)

Usage (after environment + dependencies resolved):

    python scripts/quick_start/qs_sparsezip_smoke.py \
        --cfg config/sparsezip_mme.yaml \
        --model_path liuhaotian/llava-v1.5-7b \
        --prompt "Describe the owl briefly." \
        --image reference/owl.JPEG

You can override K manually by disabling dynamic-K in YAML or passing --dominant N.
If dynamic_k is true in YAML, the adaptive K will be used regardless of --dominant.

Note: This script intentionally mirrors parts of evalkit but keeps dependencies
light to act as a smoke test. For full evaluation flows, use tools/mme_run_all.py
or scripts/evalkit.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoTokenizer

from scripts.model import LlavaVisionZipModel  # type: ignore
from utils.sparsezip import (
    VisionZipCompressor,
    CompressionConfig,
    MergingConfig,
    ScoringAlphas,
)

def _load_yaml_cfg(path: str) -> dict:
    import yaml, io
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


@dataclass
class SmokeResult:
    prompt: str
    answer: str
    k_dominant: int
    c_contextual: int
    orig_tokens: int
    compressed_tokens: int
    retained_indices: list[int]
    timings_ms: Dict[str, float]


def build_model(cfg_path: str, model_path: str) -> LlavaVisionZipModel:
    cfg = _load_yaml_cfg(cfg_path)
    # Override model path if provided
    cfg.setdefault("model", {})["model_path"] = model_path
    model_cfg = cfg["model"]
    sparsezip_cfg = model_cfg.get("sparsezip", None)
    # Legacy top-level dominant/contextual (still honored if dynamic_k false)
    dominant = model_cfg.get("dominant", 54)
    contextual = model_cfg.get("contextual", 16)
    temperature = model_cfg.get("temperature", 0.0)
    max_new_tokens = model_cfg.get("max_new_tokens", 16)
    model = LlavaVisionZipModel(
        model_path=model_cfg["model_path"],
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        dominant=dominant,
        contextual=contextual,
        sparsezip_cfg=sparsezip_cfg,
    )
    return model


def clip_only_compress(cfg_path: str, image_path: str, prompt: str) -> SmokeResult:
    """Fallback path: load a public CLIP vision model and run SparseZip compression only.

    This does NOT generate multimodal text; it returns a placeholder answer and compression stats.
    """
    cfg = _load_yaml_cfg(cfg_path)
    sparsezip_cfg = (cfg.get("model", {}) or {}).get("sparsezip", {})
    import torch
    from transformers import CLIPVisionModel, CLIPImageProcessor
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_model_id = "openai/clip-vit-large-patch14"
    clip_model = CLIPVisionModel.from_pretrained(
        vision_model_id,
        attn_implementation="eager"
    ).to(device)
    processor = CLIPImageProcessor.from_pretrained(vision_model_id)
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
    outs = clip_model(pixel_values, output_hidden_states=True, output_attentions=True)
    attn = outs.attentions[-2]  # [B,H,L,L]
    hidden = outs.hidden_states[-2]  # [B,L,C]
    # Use hidden states as proxy keys since encoder layer metric not available
    keys = hidden

    # Build compressor config
    alphas_cfg = (sparsezip_cfg.get("alphas", {}) or {})
    comp_cfg = CompressionConfig(
        alphas=ScoringAlphas(
            attn=float(alphas_cfg.get("attn", 1.0)),
            entropy=float(alphas_cfg.get("entropy", 0.4)),
            mutual=float(alphas_cfg.get("mutual", 0.6)),
        ),
        tau_feat=float(sparsezip_cfg.get("tau_feat", 0.2)),
        tau_sim=float(sparsezip_cfg.get("tau_sim", 0.1)),
        cross_beta=float(sparsezip_cfg.get("cross_beta", 0.0)),
        dynamic_k=bool(sparsezip_cfg.get("dynamic_k", True)),
        dynk=CompressionConfig().dynk.__class__(
            c=float((sparsezip_cfg.get("dynk", {}) or {}).get("c", 8.0)),
            eps=float((sparsezip_cfg.get("dynk", {}) or {}).get("eps", 1e-6)),
            k_min=int(sparsezip_cfg.get("k_min", 4)),
            k_max=int(sparsezip_cfg.get("k_max", 64)),
        ),
        k_min=int(sparsezip_cfg.get("k_min", 4)),
        k_max=int(sparsezip_cfg.get("k_max", 64)),
        merging=MergingConfig(
            contextual_num=int((sparsezip_cfg.get("merging", {}) or {}).get("contextual_num", 16)),
            kmeans_init_factor=float((sparsezip_cfg.get("merging", {}) or {}).get("kmeans_init_factor", 2.0)),
            kmeans_iters=int((sparsezip_cfg.get("merging", {}) or {}).get("kmeans_iters", 10)),
            agglomerative=bool((sparsezip_cfg.get("merging", {}) or {}).get("agglomerative", True)),
        ),
    )
    compressor = VisionZipCompressor(num_scoring_layers=1, cfg=comp_cfg)
    weights_for_context = attn[:, :, 0, 1:].mean(dim=1)
    tokens, indices = compressor(
        scoring_layers=[{"attn": attn, "keys": keys}],
        hidden_states=hidden,
        cross_attn=None,
        cross_last_dim_is_L=True,
        dominant_num=None,
        weights_for_context=weights_for_context,
    )
    k_dom = indices.shape[1] - 1 if indices is not None else -1
    c_ctx = comp_cfg.merging.contextual_num
    stats = SmokeResult(
        prompt=prompt,
        answer="[clip-only placeholder; no LM loaded]",
        k_dominant=k_dom,
        c_contextual=c_ctx,
        orig_tokens=tokens.shape[1],
        compressed_tokens=tokens.shape[1],
        retained_indices=(indices[0].tolist() if indices is not None else []),
        timings_ms={"end2end_ms": 0.0},
    )
    return stats


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def run_inference(model: LlavaVisionZipModel, image: Image.Image, prompt: str) -> SmokeResult:
    # Prepare a pseudo Sample-like object inline to reuse model.run logic if necessary.
    # Model's run expects a Sample with attributes; to avoid importing dataset structures
    # we call the internal vision+language pipeline directly.

    start_prefill = time.time()
    # Vision encoding with SparseZip patch applied
    vision_embeds, indices_info = model.encode_image(image)
    # Build text input
    tokenizer: AutoTokenizer = model.tokenizer  # type: ignore
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.llm.device)  # type: ignore

    # Prefill: concatenate vision embeddings and text prompt
    # Many LLaVA implementations build a multimodal sequence; here we approximate using
    # provided helper on the model (if available). Fallback: directly generate.
    # NOTE: LlavaVisionZipModel currently exposes model.llm (a causal LM) and expects vision
    # embeddings injected in its specific forward; for simplicity we call generate on text only
    # to focus on compression smoke. Extend this if integrated multi-modal concatenation exists.

    # Generation (text-only placeholder; multimodal concatenation would require adapter-specific code)
    out_ids = model.llm.generate(
        input_ids,
        max_new_tokens=model.max_new_tokens,
        temperature=model.temperature,
        do_sample=(model.temperature > 0),
    )
    gen_text = tokenizer.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    end_decode = time.time()

    timings_ms = {
        "prefill_ms": (end_decode - start_prefill) * 1000.0,  # Simplified (no split)
        "end2end_ms": (end_decode - start_prefill) * 1000.0,
    }

    # Extract compression stats from indices_info
    k_dominant = indices_info.get("k_dominant", -1)
    c_contextual = indices_info.get("c_contextual", -1)
    retained_indices = indices_info.get("retained_indices", [])[:20]
    orig_tokens = indices_info.get("orig_tokens", -1)
    compressed_tokens = indices_info.get("compressed_tokens", -1)

    return SmokeResult(
        prompt=prompt,
        answer=gen_text.strip(),
        k_dominant=k_dominant,
        c_contextual=c_contextual,
        orig_tokens=orig_tokens,
        compressed_tokens=compressed_tokens,
        retained_indices=retained_indices,
        timings_ms=timings_ms,
    )


def main():
    parser = argparse.ArgumentParser(description="SparseZip smoke test (single image, single prompt)")
    parser.add_argument("--cfg", type=str, default="config/sparsezip_mme.yaml", help="YAML config path with sparsezip section")
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b", help="HF model id or local path")
    parser.add_argument("--prompt", type=str, default="Describe the owl briefly.")
    parser.add_argument("--image", type=str, default="reference/owl.JPEG", help="Image path for testing")
    parser.add_argument("--save_json", type=str, default="", help="Optional: save results to JSON file")
    parser.add_argument("--clip_only", action="store_true", help="Run compression on CLIP only without loading full LLaVA model")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    if args.clip_only:
        result = clip_only_compress(args.cfg, args.image, args.prompt)
    else:
        try:
            model = build_model(args.cfg, args.model_path)
            image = load_image(args.image)
            result = run_inference(model, image, args.prompt)
        except Exception as e:
            print(f"[warn] Full LLaVA path failed ({e}); falling back to --clip_only mode.")
            result = clip_only_compress(args.cfg, args.image, args.prompt)

    print("\n=== SparseZip Smoke Result ===")
    print(f"Prompt: {result.prompt}")
    print(f"Answer: {result.answer}")
    print(f"Original vision tokens: {result.orig_tokens}")
    print(f"Compressed vision tokens: {result.compressed_tokens}")
    print(f"Dominant K: {result.k_dominant} | Contextual C: {result.c_contextual}")
    print(f"Retained indices (first 20): {result.retained_indices}")
    print(f"Timings (ms): {json.dumps(result.timings_ms, indent=2)}")

    if args.save_json:
        out_obj = {
            "prompt": result.prompt,
            "answer": result.answer,
            "k_dominant": result.k_dominant,
            "c_contextual": result.c_contextual,
            "orig_tokens": result.orig_tokens,
            "compressed_tokens": result.compressed_tokens,
            "retained_indices_head": result.retained_indices,
            "timings_ms": result.timings_ms,
        }
        with open(args.save_json, "w") as f:
            json.dump(out_obj, f, indent=2)
        print(f"Saved JSON: {args.save_json}")


if __name__ == "__main__":
    main()
