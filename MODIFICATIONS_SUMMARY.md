# SparseZip Codebase Modifications Summary

## Overview

This document summarizes all modifications made to the VisionZip-exp codebase during the debugging, evaluation, and analysis process for SparseZip performance assessment.

**Timeline**: Initial setup → Bug fixes → VM deployment → Full evaluation → Performance analysis

**Outcome**: Successfully enabled SparseZip evaluation but discovered **performance degradation** compared to baseline (not improvement).

---

## 1. Critical Bug Fixes

### 1.1 CUDA Syntax Error in `modelling_sparse_llama.py`

**File**: `models/SparseVLMs/llava/model/language_model/modelling_sparse_llama.py`  
**Line**: ~1749-1751  
**Issue**: Runtime CUDA error with `next_tokens.tile(...).ne(...).prod(...)` operation

**Original Code**:
```python
unfinished_sequences = unfinished_sequences.mul(
    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
)
```

**Fixed Code**:
```python
# Simplified EOS check
is_eos = torch.isin(next_tokens, eos_token_id_tensor)
unfinished_sequences = unfinished_sequences.mul((~is_eos).long())
```

**Impact**: ✅ **Critical fix** - Enabled model to complete generation without crashing. This was blocking ALL evaluations.

---

## 2. Attention Implementation Changes

### 2.1 Force SDPA Attention in `builder.py`

**File**: `models/SparseVLMs/llava/model/builder.py`  
**Lines**: 109-114

**Modification**:
```python
# Force SDPA attention implementation
if "attn_implementation" in kwargs:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = kwargs["attn_implementation"]
    kwargs["config"] = config
```

**Reason**: SparseVLM models require SDPA (Scaled Dot Product Attention) instead of flash_attention_2 for compatibility with sparse layer implementation.

**Impact**: ✅ Fixed model initialization errors.

### 2.2 Set Attention Implementation in `model.py`

**File**: `scripts/model.py`  
**Line**: 373

**Modification**:
```python
extra_kwargs = {
    "device_map": None if is_macos else "auto",
    "torch_dtype": torch.float16,
    "attn_implementation": "sdpa",  # Required by SparseVLM
}
```

**Impact**: ✅ Ensured consistent attention mechanism across all model loads.

---

## 3. Vision Tower Output Handling

### 3.1 Handle Tuple Return in `llava_arch.py`

**File**: `models/SparseVLMs/llava/model/llava_arch.py`  
**Lines**: 138-144

**Modification**:
```python
def encode_images(self, images):
    image_features = self.get_model().get_vision_tower()(images)
    # Handle tuple return from SparseZip vision tower (features, indices)
    if isinstance(image_features, tuple):
        image_features = image_features[0]
    image_features = self.get_model().mm_projector(image_features)
    return image_features
```

**Reason**: SparseZip's vision tower returns `(compressed_features, token_indices)` tuple, but downstream code expects only features.

**Impact**: ✅ Fixed shape mismatch errors during image encoding.

---

## 4. Model Loading & Deployment

### 4.1 Hugging Face Model Fallback in `model.py`

**File**: `scripts/model.py`  
**Lines**: 367-420

**Modifications**:
- Added automatic model download from Hugging Face Hub
- Implemented fallback quantization attempts (4bit → 8bit → fp16)
- Added macOS-specific handling (no bitsandbytes support)

**Impact**: ✅ Enabled evaluation without requiring local model files.

### 4.2 Environment Setup for VM

**Created**: Multiple fix scripts (`fix_all.sh`, `fix_syntax.py`, `download_mme_v*.sh`)

**Key Changes**:
- Set `HF_HOME` to data disk to avoid boot disk space issues
- Set `PYTHONPATH` to include project root for module imports
- Removed broken conda activation (used existing env)

**Impact**: ✅ Resolved disk space and import errors on GCP VM.

---

## 5. Evaluation Infrastructure

### 5.1 MME Evaluation Script

**Created**: `run_mme_eval.sh`

**Features**:
- Runs evaluation for 4 models: LLaVA, VisionZip, SparseVLM, SparseZip
- Uses correct MME dataset path (`datasets/mme/MME_Benchmark_release_version/MME_Benchmark`)
- Aggregates results into `summary_final.csv`

### 5.2 Configuration Files

**Created**: `config/default.yaml`

```yaml
dataset: mme
model_type: llava_vzip
temperature: 0.2
max_new_tokens: 128
```

**Purpose**: Provides baseline configuration for non-SparseZip models (required by `mme_run_all.py`).

---

## 6. Performance Analysis

### 6.1 Comprehensive Results Document

**Created**: `EVALUATION_RESULTS.md` (pushed to GitHub)

**Contents**:
- MME benchmark: 10 subtasks × 4 models
- POPE benchmark: 3 variants × 4 models
- Accuracy, latency, F1/precision/recall comparisons
- Production deployment recommendations

---

## 7. What We Did NOT Change

### 7.1 Core SparseZip Algorithm

**No modifications** were made to:
- `utils/sparsezip.py` - Token compression logic
- Vision tower forward pass
- Scoring mechanisms (attention, entropy, mutual information)
- Merging strategies (k-means, hierarchical)

**Reason**: We focused on making the existing implementation runnable, not optimizing it.

### 7.2 Hyperparameters

**Default values** were used throughout:
- `dominant_num`: Dynamic (k-means determined)
- `contextual_num`: 10-16 (from config)
- `tau_feat`: 0.2
- `tau_sim`: 0.1
- `alphas`: {attn: 1.0, entropy: 0.4, mutual: 0.6}

**Impact**: ⚠️ No hyperparameter tuning was performed, which may partially explain poor performance.

---

## 8. Performance Impact Summary

### 8.1 Did Performance Improve?

**Short Answer**: ❌ **No, SparseZip underperformed baseline.**

### 8.2 Before vs. After Modifications

**Before Modifications**:
- ❌ SparseZip: Non-functional (CUDA errors, syntax errors)
- ❌ Cannot complete evaluation

**After Modifications**:
- ✅ SparseZip: Functional and runnable
- ✅ Full evaluation completed
- ❌ **But: 8-9% accuracy degradation, minimal speedup**

### 8.3 Final Performance Metrics

| Benchmark | Baseline (LLaVA) | SparseZip | Change |
|-----------|------------------|-----------|--------|
| **MME Accuracy** | 76.2% | 67.4% | **-8.8%** ⚠️ |
| **MME Latency** | 219ms | 236ms | **+7.8%** ⚠️ (slower!) |
| **POPE Accuracy** | 85.0% | 76.4% | **-8.6%** ⚠️ |
| **POPE Latency** | 642ms | 269ms | **-58%** ✅ (faster) |

**Key Finding**: SparseZip is **slower** on MME and **less accurate** on both benchmarks. Only POPE latency improved significantly.

---

## 9. Root Cause Analysis

### 9.1 Why Did SparseZip Underperform?

**Hypothesis 1: Overly Aggressive Compression**
- Dynamic k-selection may be compressing too many tokens
- Loss of critical visual information for complex reasoning tasks

**Hypothesis 2: MME-Specific Challenges**
- MME tasks (OCR, counting, code reasoning) may require dense visual features
- Compression hurts fine-grained perception

**Hypothesis 3: Unoptimized Hyperparameters**
- Default `tau_feat`, `tau_sim`, `alphas` may not be optimal
- `contextual_num` may be too low (10-16 vs. potential 30-50)

### 9.2 Why Was POPE Latency Fast?

**Hypothesis**: POPE is a simpler yes/no task that benefits from:
- Lower token count reduces generation overhead
- Less complex reasoning required
- Binary decision making tolerates compression better

---

## 10. Recommendations for Future Work

### 10.1 Immediate Next Steps

1. **Hyperparameter Tuning**:
   - Sweep `contextual_num`: [10, 20, 30, 40, 50]
   - Tune `tau_feat` and `tau_sim` for less aggressive compression
   - Adjust `alphas` to prioritize attention-based scoring

2. **Diagnostic Analysis**:
   - Log actual token retention counts per image
   - Visualize which tokens are being discarded
   - Compare compressed vs. full features on specific MME subtasks

3. **Selective Compression**:
   - Use dynamic compression based on task complexity
   - Preserve more tokens for OCR, counting, code reasoning
   - Aggressive compression only for scene/landmark recognition

### 10.2 Long-Term Optimizations

1. **Learned Compression**:
   - Fine-tune compression parameters on MME training set
   - Task-adaptive token selection

2. **Hybrid Architecture**:
   - Combine SparseZip with VisionZip (which showed better results)
   - Use VisionZip for general compression, SparseZip for specific tasks

3. **Inference Optimization**:
   - Profile where time is spent (compression vs. generation)
   - Parallelize k-means clustering
   - Optimize CUDA kernels for token merging

---

## 11. Lessons Learned

### 11.1 Technical Lessons

1. **SDPA vs. Flash Attention**: SparseVLM requires SDPA for compatibility - flash_attention_2 causes runtime errors.

2. **Tuple Returns**: Vision towers with compression return `(features, indices)` tuples - always check return types.

3. **Git Authentication**: GCP VMs require PAT tokens for GitHub pushes - password auth is deprecated.

4. **Disk Space Management**: HuggingFace cache can fill boot disk - always set `HF_HOME` to data disk on cloud VMs.

### 11.2 Research Lessons

1. **Compression ≠ Speedup**: Token compression doesn't always translate to latency reduction (see MME results).

2. **Task Sensitivity**: Different benchmarks respond differently to compression (POPE: fast, MME: slow).

3. **Accuracy-Speed Trade-offs**: VisionZip (19% faster, 3% accuracy loss) >> SparseZip (7.8% slower, 8.8% accuracy loss).

---

## 12. Files Modified (Complete List)

### Core Model Files
1. `models/SparseVLMs/llava/model/language_model/modelling_sparse_llama.py` - CUDA fix
2. `models/SparseVLMs/llava/model/builder.py` - SDPA config
3. `models/SparseVLMs/llava/model/llava_arch.py` - Tuple handling
4. `scripts/model.py` - HF fallback, SDPA setting

### Evaluation Scripts (Created)
5. `run_mme_eval.sh` - MME orchestration
6. `run_full_evaluation.sh` - Full pipeline
7. `config/default.yaml` - Default config
8. `fix_syntax.py` - Syntax error repair
9. `fix_all.sh`, `fix_path.sh`, `download_mme_v*.sh` - VM setup

### Documentation (Created)
10. `EVALUATION_RESULTS.md` - Comprehensive analysis (pushed to GitHub)

---

## Conclusion

**What We Achieved**:
✅ Fixed critical bugs preventing SparseZip from running  
✅ Successfully completed full MME and POPE evaluations  
✅ Established evaluation infrastructure for future experiments  
✅ Identified performance gaps and root causes  

**What We Did NOT Achieve**:
❌ Performance improvement over baseline  
❌ Competitive accuracy-speed trade-off  
❌ Hyperparameter optimization  

**Next Steps**: Focus on hyperparameter tuning and diagnostic analysis to understand why compression hurts accuracy more than it helps speed.
