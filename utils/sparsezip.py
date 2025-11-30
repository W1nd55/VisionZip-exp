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
        p = _safe_softmax(x, dim=-1, tau=self.tau_feat, eps=self.eps)
        H = -(p * p.log()).sum(dim=-1) / math.log(x.shape[-1] + self.eps)
        z = _normalize_l2(x, dim=-1, eps=self.eps)
        sim = torch.bmm(z, z.transpose(1, 2))
        eye = torch.eye(sim.size(-1), device=sim.device).bool().unsqueeze(0)
        sim = sim.masked_fill(eye, -1e9)
        q = _safe_softmax(sim, dim=-1, tau=self.tau_sim, eps=self.eps)
        Hsim = -(q * q.log()).sum(dim=-1) / math.log(q.shape[-1] + self.eps)
        I = 1.0 - Hsim
        s1 = _zscore(s_attn, dim=1, eps=self.eps)
        s2 = _zscore(H, dim=1, eps=self.eps)
        s3 = _zscore(I, dim=1, eps=self.eps)
        a = self.alphas
        return a.attn * s1 + a.entropy * s2 + a.mutual * s3

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
    var = scores.var(dim=1, unbiased=False)
    k = torch.log(var + cfg.eps) + cfg.c
    k = torch.round(k).clamp(min=cfg.k_min, max=cfg.k_max).to(torch.int64)
    return torch.clamp(k, min=1)

# -------- Hierarchical Merging ------- #

@dataclass
class MergingConfig:
    contextual_num: int = 16
    kmeans_init_factor: float = 2.0
    kmeans_iters: int = 10
    agglomerative: bool = True
    eps: float = 1e-12


def _kmeans_one(x: torch.Tensor, k: int, w: Optional[torch.Tensor], iters: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    device = x.device
    N, D = x.shape
    k = min(max(1, k), N)
    idx0 = torch.randint(0, N, (1,), device=device)
    cents = [x[idx0]]
    if k > 1:
        d2 = torch.cdist(x, cents[0], p=2).squeeze(-1) ** 2
        for _ in range(1, k):
            probs = d2 / (d2.sum() + 1e-12)
            nxt = torch.multinomial(probs.clamp_min(1e-12), 1)
            cents.append(x[nxt])
            d2 = torch.minimum(d2, (torch.cdist(x, cents[-1], p=2).squeeze(-1) ** 2))
    centroids = torch.cat(cents, dim=0)

    assign = torch.zeros(N, dtype=torch.long, device=device)
    for _ in range(iters):
        dist = torch.cdist(x, centroids, p=2)
        assign = dist.argmin(dim=1)
        for ci in range(centroids.shape[0]):
            mask = (assign == ci)
            if mask.any():
                if w is not None:
                    ww = w[mask].unsqueeze(-1)
                    centroids[ci] = (ww * x[mask]).sum(dim=0) / (ww.sum() + 1e-12)
                else:
                    centroids[ci] = x[mask].mean(dim=0)
            else:
                ridx = torch.randint(0, N, (1,), device=device)
                centroids[ci] = x[ridx]
    return centroids, assign


def _agglomerative_merge(centroids: torch.Tensor, target_k: int) -> torch.Tensor:
    while centroids.shape[0] > target_k:
        z = _normalize_l2(centroids, dim=-1)
        sim = torch.matmul(z, z.T)
        sim.fill_diagonal_(-1.0)
        i = torch.argmax(sim).item() // sim.shape[1]
        j = torch.argmax(sim).item() % sim.shape[1]
        if i > j:
            i, j = j, i
        merged = (centroids[i] + centroids[j]) / 2.0
        centroids = torch.cat([centroids[:i], centroids[i+1:j], centroids[j+1:], merged.unsqueeze(0)], dim=0)
    return centroids


def hierarchical_context_merge(
    hidden_remain: torch.Tensor,
    keys_remain: torch.Tensor,
    weights_remain: Optional[torch.Tensor],
    cfg: MergingConfig,
) -> torch.Tensor:
    B, Nr, C = hidden_remain.shape
    z_all = _normalize_l2(keys_remain.float(), dim=-1)
    contextual_tokens = []
    for b in range(B):
        z = z_all[b]
        h = hidden_remain[b]
        w = None if weights_remain is None else weights_remain[b]
        init_k = min(int(max(cfg.contextual_num, math.ceil(cfg.kmeans_init_factor * cfg.contextual_num))), Nr)
        cents, assign = _kmeans_one(z, init_k, w, iters=cfg.kmeans_iters)
        if cfg.agglomerative and init_k > cfg.contextual_num:
            cents = _agglomerative_merge(cents, cfg.contextual_num)
        dist = torch.cdist(z, cents, p=2)
        final_ids = dist.argmin(dim=1)
        ctoks = []
        for ci in range(cfg.contextual_num):
            mask = (final_ids == ci)
            if not mask.any():
                ctoks.append(torch.zeros(C, device=h.device, dtype=h.dtype))
                continue
            if w is not None:
                ww = w[mask].unsqueeze(-1)
                agg = (ww * h[mask]).sum(dim=0) / (ww.sum() + cfg.eps)
            else:
                agg = h[mask].mean(dim=0)
            ctoks.append(agg)
        contextual_tokens.append(torch.stack(ctoks, dim=0))
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
                w_rem = None
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
