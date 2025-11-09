# SparseZip: Dynamic Vision Token Compression

Updated: 2025-11-08

This document explains the SparseZip compressor we added to the codebase, how it works, where the code lives, what changed in the pipeline, and how to run it.

## What is SparseZip?

SparseZip is a fast, text-aware vision token compression module for LLaVA-like MLLMs. It reduces the number of vision tokens fed into the LLM by:

- Selecting K dominant tokens by a hybrid importance score (attention + entropy + mutual-information proxy)
- Dynamically adapting K per image using the variance of scores (Dynamic-K)
- Merging non-dominant tokens into C contextual tokens using hierarchical clustering with attention-weighted aggregation
- Optionally fusing cross-attention information (API ready)
- Optionally fusing scores across multiple vision depths with learned gating

Goal: preserve key image content while shrinking sequence length to speed up inference.

## Code locations

- Core compressor: `utils/sparsezip.py`
  - `LayerwiseHybridScorer`: computes hybrid scores per patch for one or more layers
  - `dynamic_k_from_scores`: implements K = round(log(var(scores)+eps)+c)
  - `hierarchical_context_merge`: k-means init + (optional) agglomerative merge + attention-weighted averaging
  - `VisionZipCompressor`: end-to-end routine returning condensed tokens + [CLS, dominant] indices

- Integration into LLaVA: `scripts/model.py`
  - In `LlavaVisionZipModel.__init__` we patch `CLIPVisionTower.forward` with `_sparsezip_forward`
  - The patch requests attentions/hidden/keys from the CLIP vision tower, runs `VisionZipCompressor`, and returns condensed tokens
  - Compressor config is taken from YAML (see below) and attached to the vision tower as `_sparsezip_cfg`

- YAML plumbing: `scripts/evalkit.py`
  - Passes `model.sparsezip` to `LlavaVisionZipModel` via `sparsezip_cfg`

- Example configs: `config/sparsezip_mme.yaml`, `config/sparsezip_pope.yaml`

## What changed (summary of code edits)

1) New module: `utils/sparsezip.py`
   - Implements: hybrid scoring, dynamic-K, hierarchical merging, multi-layer gating, cross-attn hook
   - Includes a small `__main__` smoke test for shapes

2) Patched model: `scripts/model.py`
   - Added `sparsezip_cfg` to `LlavaVisionZipModel` constructor
   - Patched `CLIPVisionTower.forward` -> `_sparsezip_forward`
   - `_sparsezip_forward` now:
     - Calls CLIP with `output_hidden_states=True, output_attentions=True`
     - Uses layer -2 attentions/hidden/keys
     - Builds/updates a `VisionZipCompressor` configured from YAML
     - Computes dominant tokens via dynamic-K hybrid scoring
     - Builds contextual tokens via hierarchical merge (attention-weighted)
     - Returns condensed tokens (dtype-matched) and indices

3) YAML wiring: `scripts/evalkit.py`
   - `build_model_llava_vzip` now forwards `model.sparsezip` as `sparsezip_cfg`

4) New YAMLs:
   - `config/sparsezip_mme.yaml`
   - `config/sparsezip_pope.yaml`

## Configuration (YAML)

In your YAML under `model:` add a `sparsezip:` section. Example (MME):

```yaml
model:
  model_type: llava_vzip
  model_path: liuhaotian/llava-v1.5-7b
  temperature: 0.0
  max_new_tokens: 16

  # For legacy compatibility; dynamic-K can still be enabled below
  dominant: 54
  contextual: 16

  sparsezip:
    dynamic_k: true           # enable adaptive K per image
    k_min: 8                  # clamp lower bound for K
    k_max: 64                 # clamp upper bound for K
    dynk:
      c: 8.0                 # bias in K = round(log(var(scores)+eps)+c)
      eps: 1.0e-6

    # scoring weights
    alphas:
      attn: 1.0
      entropy: 0.4
      mutual: 0.6
    tau_feat: 0.2            # feature softmax temperature
    tau_sim: 0.1             # similarity softmax temperature

    cross_beta: 0.0          # cross-attn fusion weight (kept 0.0 by default)

    # contextual merging
    merging:
      contextual_num: 16
      kmeans_init_factor: 2.0
      kmeans_iters: 10
      agglomerative: true
```

## How to run

- MME (run all subtasks):
```bash
python tools/mme_run_all.py \
  --mme_root /path/to/MME_Benchmark \
  --out_root ./runs/mme_sparsezip \
  --cfg config/sparsezip_mme.yaml \
  --model_path liuhaotian/llava-v1.5-7b
```

- POPE:
```bash
python tools/pope_run_all.py \
  --cfg config/sparsezip_pope.yaml \
  --model_path liuhaotian/llava-v1.5-7b \
  --out_dir ./runs/pope_sparsezip
```

- Single run via evalkit (any dataset):
```bash
python scripts/evalkit.py --cfg config/sparsezip_mme.yaml --ann_path PATH/TO/ANN.json --output_dir ./runs/test
```

### Smoke test (compression only or full generation)

Use the quick-start script on a single image (owl.JPEG provided in `reference/`).

Full attempt (will try to load whole LLaVA; on macOS may fall back to clip-only if load fails):
```bash
python scripts/quick_start/qs_sparsezip_smoke.py \
  --cfg config/sparsezip_mme.yaml \
  --image reference/owl.JPEG \
  --prompt "Describe the owl briefly." 
```

Force clip-only compression mode (bypasses LLM):
```bash
python scripts/quick_start/qs_sparsezip_smoke.py \
  --cfg config/sparsezip_mme.yaml \
  --image reference/owl.JPEG \
  --prompt "Describe the owl briefly." \
  --clip_only
```

Expected output prints: dynamic K, contextual C, indices of retained dominant tokens.

### Fixed-K configuration example

Disable dynamic-K and set dominant count explicitly:
```yaml
model:
  dominant: 48     # fixed number of dominant tokens
  contextual: 16
  sparsezip:
    dynamic_k: false
    k_min: 8
    k_max: 64
    alphas:
      attn: 1.0
      entropy: 0.4
      mutual: 0.6
    merging:
      contextual_num: 16
```

### Dynamic-K logging snippet

If you want to log K during runs, add inside `_sparsezip_forward` after compression:
```python
print(f"[SparseZip] dynamic K={all_indices.shape[1]-1} C={self._vz_comp.cfg.merging.contextual_num}")
```

### Disabling contextual merging

To keep only dominant tokens (no contextual aggregation), set:
```yaml
model:
  sparsezip:
    merging:
      contextual_num: 0   # or 1 if you need at least one aggregate
```
The compressor will skip merge logic and return CLS + K dominant tokens only.

### Multi-layer scoring (advanced)

Current integration uses layer -2. To experiment with multi-layer fusion:
1. Modify `_sparsezip_forward` to collect several `attn_weights` and `hidden_states` layers.
2. Pass a list to `scoring_layers=[{"attn": attn_l_i, "keys": keys_l_i}, ...]` when calling the compressor.
3. Increase `num_scoring_layers` in `VisionZipCompressor` constructor.

### Clip-only mode rationale

When full model load is heavy (VRAM limits or macOS fp16 issues), `--clip_only` allows verifying compression logic (scores, dynamic-K) without loading the LLM weights. It exercises CLIP forward + SparseZip only.

### macOS notes

- bitsandbytes 4/8-bit quantization is typically unavailable; loader falls back to fp16 then fp32.
- Avoid `device_map="auto"` on smaller GPUs to prevent fragmented CPU offload issues; the patched loader now sets `device_map=None` on macOS.
- Install `protobuf` to avoid tokenizer serialization errors.

## Design notes & tips

- Dynamic-K vs Fixed-K: For fixed-K experiments, set `sparsezip.dynamic_k: false` and choose a numeric `model.dominant` value. For dynamic-K, set `dynamic_k: true`.
- Contextual tokens: `merging.contextual_num` controls compression of the non-dominant tokens; increase for more context.
- Cross-attention: The compressor supports fusing cross-attn with `cross_beta>0`, but capturing LLM cross-attn requires extra wiring (not enabled by default to keep the vision tower patch self-contained).
- Multi-layer scoring: To use multi-layer scoring, feed multiple layers into the compressor. The current integration samples layer -2 for simplicity.
- Performance: Similarity matrix is O((L-1)^2); consider reducing vision resolution if memory is tight.

## Troubleshooting

- “Import not resolved” in editor: These are environment warnings in the editor for PyTorch/LLaVA; runtime with proper env is fine.
- No change with YAML: Ensure you used the `sparsezip_mme.yaml` or `sparsezip_pope.yaml`. The compressor reads config from `model.sparsezip`.
- Fallback path used: If the SparseZip patch fails, the code falls back to the older EXP forward; check console warnings.
- Dynamic-K always returns same K: Ensure variance isn't degenerate; very uniform images may clamp near `k_min`. Try increasing `alphas.entropy` or lowering `dynk.c`.
- Memory spikes during merge: Reduce `kmeans_init_factor` or disable `agglomerative`.

## Changelog (SparseZip)

- Added `utils/sparsezip.py`
- Patched `scripts/model.py` to `_sparsezip_forward`, added `sparsezip_cfg`
- `scripts/evalkit.py` forwards YAML `model.sparsezip` to the model
- Added YAMLs: `config/sparsezip_mme.yaml`, `config/sparsezip_pope.yaml`
