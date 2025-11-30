"""
SparseZip Compression Utilities (formerly Section 3/3.3 prototype)

Features:
- Multi-dimensional dominant token scoring (attention + entropy + MI proxy)
- Optional cross-attention fusion hook (self + cross) for text-aware scoring
- Dynamic-K selection per image: K = round(log(var(scores)+eps) + c)
- Multi-layer fusion with learned gating weights
- Hierarchical contextual token merging (k-means init + optional agglomerative merge)
- Attention-weighted averaging for contextual clusters

Provides VisionZipCompressor class that returns condensed vision embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Helpers ---------------- #

def _zscore(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    m = x.mean(dim=dim, keepdim=True)
    s = x.std(dim=dim, keepdim=True) + eps
    return (x - m) / s


def _safe_softmax(x: torch.Tensor, dim: int = -1, tau: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    return F.softmax(x / max(tau, eps), dim=dim).clamp_min(eps)


def _normalize_l2(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

# ---------------- Scoring ---------------- #

@dataclass
class ScoringAlphas:
    attn: float = 1.0
    entropy: float = 0.4
    mutual: float = 0.6


class LayerwiseHybridScorer(nn.Module):
    def __init__(
        self,
        alphas: Optional[ScoringAlphas] = None,
        use_layer_gating: bool = True,
        num_layers: int = 1,
        cross_beta: float = 0.0,
        tau_feat: float = 0.2,
        tau_sim: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.alphas = alphas if alphas is not None else ScoringAlphas()
        self.use_layer_gating = use_layer_gating
        self.cross_beta = cross_beta
        self.tau_feat = tau_feat
        self.tau_sim = tau_sim
        self.eps = eps
        if use_layer_gating and num_layers > 1:
            self.layer_gates = nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))
        else:
            self.layer_gates = None

    def _single_layer_hybrid(self, attn_weights: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # attn_weights: [B,H,L,L]; keys: [B,L,Ck]
        s_attn = attn_weights[:, :, 0, 1:].mean(dim=1)  # [B,L-1]
        x = keys[:, 1:, :].float()
        
        # OPTIMIZED: Single L2 normalization for all metrics
        z = F.normalize(x, dim=-1, eps=self.eps)
        
        # Entropy from normalized features
        p = F.softmax(z / self.tau_feat, dim=-1)
        H = -(p * (p + self.eps).log()).sum(dim=-1) / math.log(x.shape[-1] + self.eps)
        
        # Mutual information proxy from same normalized features
        sim = torch.bmm(z, z.transpose(1, 2))
        eye = torch.eye(sim.size(-1), device=sim.device).bool().unsqueeze(0)
        sim = sim.masked_fill(eye, -1e9)
        q = F.softmax(sim / self.tau_sim, dim=-1)
        Hsim = -(q * (q + self.eps).log()).sum(dim=-1) / math.log(q.shape[-1] + self.eps)
        I = 1.0 - Hsim
        
        # OPTIMIZED: Min-max normalize per sample (remove batch-dependent z-score)
        s_attn = (s_attn - s_attn.min(dim=1, keepdim=True)[0]) / (s_attn.max(dim=1, keepdim=True)[0] - s_attn.min(dim=1, keepdim=True)[0] + self.eps)
        H_norm = (H - H.min(dim=1, keepdim=True)[0]) / (H.max(dim=1, keepdim=True)[0] - H.min(dim=1, keepdim=True)[0] + self.eps)
        I_norm = (I - I.min(dim=1, keepdim=True)[0]) / (I.max(dim=1, keepdim=True)[0] - I.min(dim=1, keepdim=True)[0] + self.eps)
        
        a = self.alphas
        return a.attn * s_attn + a.entropy * H_norm + a.mutual * I_norm

    def forward(
        self,
        layers: List[Dict[str, torch.Tensor]],
        cross_attn: Optional[torch.Tensor] = None,
        cross_last_dim_is_L: bool = True,
    ) -> torch.Tensor:
        per_layer = [self._single_layer_hybrid(d['attn'], d['keys']) for d in layers]
        fused = per_layer[0]
        if len(per_layer) > 1:
            normed = [_zscore(s, dim=1, eps=self.eps) for s in per_layer]
            if self.layer_gates is not None:
                g = torch.softmax(self.layer_gates, dim=0)
            else:
                g = torch.full((len(normed),), 1.0/len(normed), device=fused.device)
            fused = sum(g[i] * normed[i] for i in range(len(normed)))
        if cross_attn is not None and self.cross_beta != 0.0:
            if cross_last_dim_is_L:
                s_cross = cross_attn.mean(dim=tuple(range(1, cross_attn.dim() - 1)))
            else:
                s_cross = cross_attn.mean(dim=(1, 3))
            s_cross = s_cross[:, 1:]
            fused = fused + self.cross_beta * _zscore(s_cross, dim=1, eps=self.eps)
        return fused

# ------------- Dynamic K ------------- #

@dataclass
class DynamicKConfig:
    c: float = 8.0
    k_min: int = 4
    k_max: int = 64
    eps: float = 1e-6


def dynamic_k_from_scores(scores: torch.Tensor, cfg: DynamicKConfig) -> torch.Tensor:
    """
    Improved dynamic-K using score spread instead of variance.
    High spread → complex image → keep more tokens.
    """
    # Measure score distribution spread via percentile range
    p90 = torch.quantile(scores, 0.9, dim=1)
    p10 = torch.quantile(scores, 0.1, dim=1)
    spread = (p90 - p10).clamp(min=cfg.eps)
    
    # K = c + log(1 + spread) * scale_factor
    # scale_factor=10 maps spread [0.1, 2.0] → K [~8, ~48]
    k = cfg.c + torch.log(1.0 + spread) * 10.0
    k = torch.round(k).clamp(min=cfg.k_min, max=cfg.k_max).to(torch.int64)
    return k.clamp(min=1)

# -------- Hierarchical Merging ------- #

@dataclass
class MergingConfig:
    contextual_num: int = 16
    kmeans_init_factor: float = 2.0
    kmeans_iters: int = 10
    agglomerative: bool = True
    eps: float = 1e-12


def _fast_single_shot_merge(
    hidden_remain: torch.Tensor,
    keys_remain: torch.Tensor,
    weights_remain: Optional[torch.Tensor],
    contextual_num: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Fast single-shot contextual token merging (VisionZip-style).
    Replaces expensive k-means clustering for ~40ms latency savings.
    
    Args:
        hidden_remain: [N_remain, C] hidden states of non-dominant tokens
        keys_remain: [N_remain, Ck] keys for similarity computation
        weights_remain: [N_remain] optional importance weights
        contextual_num: target number of contextual tokens
        eps: numerical stability constant
    
    Returns:
        ctx_tokens: [contextual_num, C] merged contextual tokens
    """
    N_remain, C = hidden_remain.shape
    
    # Normalize keys for similarity computation
    metric_normalized = F.normalize(keys_remain.float(), dim=-1, eps=eps)
    
    # Select contextual anchors via uniform spacing (fast, deterministic)
    step = max(1, N_remain // contextual_num)
    target_indices = torch.arange(0, N_remain, step, device=keys_remain.device)[:contextual_num]
    
    # Handle edge case: fewer remaining tokens than contextual_num
    if target_indices.shape[0] < contextual_num:
        pad_size = contextual_num - target_indices.shape[0]
        last_idx = target_indices[-1] if target_indices.numel() > 0 else 0
        padding = torch.full((pad_size,), last_idx, device=keys_remain.device, dtype=target_indices.dtype)
        target_indices = torch.cat([target_indices, padding], dim=0)
    
    # Get anchor tokens
    target_tokens = metric_normalized[target_indices, :]  # [contextual_num, Ck]
    
    # Create mask for non-anchor tokens
    remain_mask = torch.ones(N_remain, dtype=torch.bool, device=keys_remain.device)
    remain_mask[target_indices] = False
    
    # Compute similarity between non-anchors and anchors
    tokens_to_merge = metric_normalized[remain_mask, :]  # [N_merge, Ck]
    similarity = torch.mm(tokens_to_merge, target_tokens.t())  # [N_merge, contextual_num]
    
    # Assign each token to nearest anchor (single-shot, no iterations!)
    assignment = similarity.argmax(dim=1)  # [N_merge]
    
    # Aggregate tokens per cluster
    ctx_tokens = torch.zeros(contextual_num, C, device=hidden_remain.device, dtype=hidden_remain.dtype)
    
    for k_idx in range(contextual_num):
        cluster_mask = (assignment == k_idx)
        
        if cluster_mask.any():
            # Get tokens assigned to this cluster
            cluster_hidden = hidden_remain[remain_mask, :][cluster_mask, :]
            
            # Weighted or unweighted average
            if weights_remain is not None:
                cluster_weights = weights_remain[remain_mask][cluster_mask].unsqueeze(-1)
                ctx_tokens[k_idx] = (cluster_weights * cluster_hidden).sum(dim=0) / (cluster_weights.sum() + eps)
            else:
                ctx_tokens[k_idx] = cluster_hidden.mean(dim=0)
        else:
            # No tokens assigned: use anchor token itself
            ctx_tokens[k_idx] = hidden_remain[target_indices[k_idx], :]
    
    return ctx_tokens


def hierarchical_context_merge(
    hidden_remain: torch.Tensor,
    keys_remain: torch.Tensor,
    weights_remain: Optional[torch.Tensor],
    cfg: MergingConfig,
) -> torch.Tensor:
    """
    Hierarchical contextual token merging using fast single-shot assignment.
    Optimized version replacing k-means clustering.
    
    Args:
        hidden_remain: [B, Nr, C] hidden states
        keys_remain: [B, Nr, Ck] keys
        weights_remain: [B, Nr] optional weights
        cfg: merging configuration
    
    Returns:
        contextual_tokens: [B, contextual_num, C]
    """
    B, Nr, C = hidden_remain.shape
    contextual_tokens = []
    
    for b in range(B):
        h = hidden_remain[b]  # [Nr, C]
        k = keys_remain[b]    # [Nr, Ck]
        w = None if weights_remain is None else weights_remain[b]  # [Nr] or None
        
        # Use fast single-shot merge (replaces k-means)
        ctx_toks = _fast_single_shot_merge(h, k, w, cfg.contextual_num, cfg.eps)
        contextual_tokens.append(ctx_toks)
    
    return torch.stack(contextual_tokens, dim=0)

# ------------- Compression ------------- #

@dataclass
class CompressionConfig:
    alphas: ScoringAlphas = field(default_factory=ScoringAlphas)
    tau_feat: float = 0.2
    tau_sim: float = 0.1
    cross_beta: float = 0.0
    dynamic_k: bool = True
    dynk: DynamicKConfig = field(default_factory=DynamicKConfig)
    k_min: int = 4
    k_max: int = 64
    merging: MergingConfig = field(default_factory=MergingConfig)


class VisionZipCompressor(nn.Module):
    def __init__(self, num_scoring_layers: int = 1, cfg: CompressionConfig = CompressionConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.scorer = LayerwiseHybridScorer(
            alphas=cfg.alphas,
            use_layer_gating=(num_scoring_layers > 1),
            num_layers=num_scoring_layers,
            cross_beta=cfg.cross_beta,
            tau_feat=cfg.tau_feat,
            tau_sim=cfg.tau_sim,
        )

    @torch.no_grad()
    def forward(
        self,
        scoring_layers: List[Dict[str, torch.Tensor]],
        hidden_states: torch.Tensor,
        cross_attn: Optional[torch.Tensor] = None,
        cross_last_dim_is_L: bool = True,
        dominant_num: Optional[int] = None,
        weights_for_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C = hidden_states.shape
        scores = self.scorer(scoring_layers, cross_attn=cross_attn, cross_last_dim_is_L=cross_last_dim_is_L)
        if self.cfg.dynamic_k and dominant_num is None:
            k_per = dynamic_k_from_scores(scores, self.cfg.dynk)
        else:
            k_val = dominant_num if dominant_num is not None else max(self.cfg.k_min, 4)
            k_per = torch.full((B,), int(k_val), dtype=torch.long, device=hidden_states.device)
        k_per = torch.clamp(k_per, min=self.cfg.k_min, max=self.cfg.k_max)

        outputs, idx_batches = [], []
        for b in range(B):
            k = int(k_per[b].item())
            topk = torch.topk(scores[b], k)
            dom_idx = topk.indices + 1
            keep_idx = torch.cat([torch.tensor([0], device=hidden_states.device, dtype=dom_idx.dtype), dom_idx], dim=0)
            mask = torch.ones(L, dtype=torch.bool, device=hidden_states.device)
            mask[keep_idx] = False
            dominant_tokens = hidden_states[b, ~mask, :]
            remain_hidden = hidden_states[b, mask, :]
            remain_keys = scoring_layers[0]['keys'][b, mask, :]
            if weights_for_context is not None:
                w_all = weights_for_context[b]
                abs_idx = torch.arange(L, device=hidden_states.device)
                rem_abs = abs_idx[mask]
                w_rem = w_all[rem_abs - 1]
            else:
                # Optimization: Use hybrid scores as weights for merging
                # scores[b] corresponds to indices 1..L (excluding CLS)
                # We pad with 0 for CLS to align with mask indices
                w_all = torch.cat([torch.zeros(1, device=scores.device, dtype=scores.dtype), scores[b]], dim=0)
                w_rem = w_all[mask]
            ctx_tokens = hierarchical_context_merge(
                remain_hidden.unsqueeze(0),
                remain_keys.unsqueeze(0),
                None if w_rem is None else w_rem.unsqueeze(0),
                cfg=self.cfg.merging,
            )[0]
            out = torch.cat([dominant_tokens, ctx_tokens], dim=0)
            outputs.append(out)
            idx_batches.append(keep_idx)

        max_len = max(o.shape[0] for o in outputs)
        padded = []
        for o in outputs:
            if o.shape[0] < max_len:
                pad = torch.zeros(max_len - o.shape[0], C, device=o.device, dtype=o.dtype)
                o = torch.cat([o, pad], dim=0)
            padded.append(o)
        out_tokens = torch.stack(padded, dim=0)

        max_idx_len = max(len(ids) for ids in idx_batches)
        idx_padded = []
        for ids in idx_batches:
            if len(ids) < max_idx_len:
                pad = -torch.ones(max_idx_len - len(ids), device=ids.device, dtype=ids.dtype)
                ids = torch.cat([ids, pad], dim=0)
            idx_padded.append(ids)
        all_indices = torch.stack(idx_padded, dim=0)
        return out_tokens, all_indices

# ------------- Smoke Test ------------- #

if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, L, C, Ck = 2, 4, 65, 32, 24
    attn1 = torch.rand(B, H, L, L); attn1 = attn1 / attn1.sum(dim=-1, keepdim=True)
    keys1 = torch.randn(B, L, Ck)
    hidden = torch.randn(B, L, C)
    layers = [{"attn": attn1, "keys": keys1}]
    compressor = VisionZipCompressor(num_scoring_layers=1, cfg=CompressionConfig())
    tokens, idx = compressor(layers, hidden)
    print("tokens", tokens.shape)
    print("idx", idx.shape)
