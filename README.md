# SparseZip: Text-Aware Visual Token Selection for Efficient VLM Inference

**CS 769 HW4 Project**

## What We're Doing

We're investigating a fundamental question in vision-language models: should visual token selection adapt to each question, or can we pick tokens once per image? We reimplemented two state-of-the-art methods—VisionZip (text-agnostic, compresses before the forward pass) and SparseVLM (text-aware, compresses during the forward pass)—to understand their trade-offs. Our proposed SparseZip framework combines the best of both: text-aware selection with compression prior to the forward pass.

## Team

- Alexander Geppert
- Qinxinghao Chen  
- Shivam Mittal
- Najib Akram Maheen Aboobacker

## What We Found

### POPE Results (6,823 samples)

Both methods perform nearly identically:
- **VisionZip**: 80.3% accuracy
- **SparseVLM**: 80.6% accuracy

For simple yes/no questions about object existence, text-aware selection doesn't help much. The paper reports 77.0% (VisionZip) and 75.1% (SparseVLM); our higher scores likely come from evaluation differences.

### MME Results (2,374 samples)

Here's where it gets interesting:
- **VisionZip**: 76.28% overall accuracy
- **SparseVLM**: 73.21% overall accuracy

VisionZip wins by 3.07%. The paper reports raw scores of 1690 (VisionZip) and 1505 (SparseVLM). Our results suggest that compressing before the forward pass avoids noise from irrelevant tokens that accumulate during progressive pruning.

### The Confounding Problem

Both methods differ in two ways: (1) text-agnostic vs text-aware selection, and (2) compression timing (before vs during forward pass). These factors are confounded—we can't tell which one drives the performance gap. That's exactly what SparseZip is designed to untangle.

## What We've Done (Phase 0 - Completed)

We've successfully reimplemented both VisionZip and SparseVLM, integrated them with LLaVA-1.5-7B, and evaluated them on POPE (6,823 samples) and MME (2,374 samples). We built evaluation pipelines, comparison scripts, and established baseline performance metrics. This foundation enables us to properly compare our proposed SparseZip method against these baselines.

## What's Next

**Phase 1 (Weeks 1-2):** Build the core SparseZip framework—hybrid attention mechanism and dynamic K selection.

**Phase 2 (Weeks 3-4):** Add experimental components: hierarchical contextual token merging and multi-dimensional scoring. Run ablation studies to see what actually helps.

**Phase 3 (Week 5):** Integrate the best components, run final evaluations on POPE and MME (maybe MMBench and ScienceQA too), and write up the results.

## Code

Main evaluation scripts:
- `scripts/eval_pope_visionzip.py` - VisionZip evaluation on POPE
- `scripts/eval_pope_sparsevlm.py` - SparseVLM evaluation on POPE
- `scripts/eval_mme_visionzip.py` - VisionZip evaluation on MME
- `scripts/eval_mme_sparsevlm.py` - SparseVLM evaluation on MME
- `scripts/compare_results.py` - Compare POPE results between methods
- `scripts/compare_mme_results.py` - Compare MME results between methods

## Report

Full details in `hw4_report.tex` (7 pages).
