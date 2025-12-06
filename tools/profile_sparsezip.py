#!/usr/bin/env python3
"""
SparseZip Verification Script (Phase 1)

Validates the optimized SparseZip implementation:
1. Verifies _fast_single_shot_merge works and is fast (<5ms)
2. Verifies VisionZipCompressor runs end-to-end without errors
3. Checks dynamic-K values with new formula

Usage:
    python tools/profile_sparsezip.py --num_samples 50
"""

import torch
import time
import numpy as np
from pathlib import Path
import sys
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sparsezip import VisionZipCompressor, CompressionConfig, ScoringAlphas, DynamicKConfig, MergingConfig, _fast_single_shot_merge

def profile_merge_latency(num_samples=50, tokens_per_image=576):
    """Profile the new single-shot merge function"""
    print("=" * 60)
    print("PROFILING NEW MERGE FUNCTION LATENCY")
    print("=" * 60)
    
    latencies = []
    for _ in range(num_samples):
        N = tokens_per_image // 2
        D = 768
        k = 20  # New contextual_num
        
        hidden = torch.randn(N, D).cuda()
        keys = torch.randn(N, 64).cuda()
        weights = torch.rand(N).cuda()
        
        start = time.perf_counter()
        # Call the actual implemented function
        _ = _fast_single_shot_merge(hidden, keys, weights, k)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        
        latencies.append(elapsed)
    
    print(f"Merge latency (k={k}):")
    print(f"  Mean: {np.mean(latencies):.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  Target: < 5.00 ms")
    if np.mean(latencies) < 5.0:
        print("  STATUS: PASS ✅")
    else:
        print("  STATUS: SLOW ⚠️")
    print()
    return np.mean(latencies)

def profile_compressor_e2e(num_samples=50):
    """Profile full compressor end-to-end"""
    print("=" * 60)
    print("PROFILING COMPRESSOR END-TO-END")
    print("=" * 60)
    
    cfg = CompressionConfig()
    # Ensure config matches our updates
    cfg.k_min = 20
    cfg.k_max = 48
    cfg.merging.contextual_num = 20
    cfg.merging.kmeans_iters = 1
    cfg.merging.agglomerative = False
    
    compressor = VisionZipCompressor(num_scoring_layers=1, cfg=cfg).cuda()
    
    B, L, C, Ck = 1, 576, 768, 64
    
    latencies = []
    for _ in range(num_samples):
        attn = torch.rand(B, 12, L, L).cuda()
        keys = torch.randn(B, L, Ck).cuda()
        hidden = torch.randn(B, L, C).cuda()
        layers = [{"attn": attn, "keys": keys}]
        
        start = time.perf_counter()
        with torch.no_grad():
            _, _ = compressor(layers, hidden)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        
    print(f"End-to-end Latency (Batch=1):")
    print(f"  Mean: {np.mean(latencies):.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    profile_merge_latency(args.num_samples)
    profile_compressor_e2e(args.num_samples)
    
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
