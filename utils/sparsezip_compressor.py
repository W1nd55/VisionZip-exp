# sparsezip_compressor.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- helpers --------- #

def _zscore(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    # Calculate Z-score (standardize) along a dimension
    m = x.mean(dim=dim, keepdim=True)
    s = x.std(dim=dim, keepdim=True) + eps
    return (x - m) / s

def _safe_softmax(
    x: torch.Tensor,
    dim: int = -1,
    tau: float = 1.0,
    eps: float = 1e-12
) -> torch.Tensor:
    # Apply temperature-scaled softmax, clamping minimum value for stability
    return F.softmax(x / max(tau, eps), dim=dim).clamp_min(eps)

def _normalize_l2(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12
) -> torch.Tensor:
    # L2 normalize tensor along a dimension
    return x / (x.norm(dim=dim, keepdim=True) + eps)

# --------- scoring --------- #

@dataclass
class ScoringAlphas:
    # Weights for the three components of the hybrid scoring metric
    attn: float = 1.0       # Attention score weight
    entropy: float = 0.4    # Feature Entropy weight
    mutual: float = 0.6     # Mutual Information proxy weight

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
        # Learnable gates for fusing scores from multiple layers
        if use_layer_gating and num_layers > 1:
            self.layer_gates = nn.Parameter(
                torch.zeros(num_layers, dtype=torch.float32)
            )
        else:
            self.layer_gates = None

    def _single_layer_hybrid(
        self,
        attn_weights: torch.Tensor,
        keys: torch.Tensor
    ) -> torch.Tensor:
        # attn_weights: [B,H,L,L]; keys: [B,L,Ck]
        # 1) CLS->patch attention (s_attn)
        # Average attention weights from CLS (index 0) to all patch tokens (index 1:) across heads
        s_attn = attn_weights[:, :, 0, 1:].mean(dim=1)  # [B,L-1]

        # 2) Feature Entropy (H)
        x = keys[:, 1:, :].float()     # [B,L-1,Ck], Patch tokens
        # Apply temperature-scaled softmax for pseudo-probability distribution
        p = _safe_softmax(x, dim=-1, tau=self.tau_feat, eps=self.eps)
        # Calculate entropy (normalized by maximum possible entropy)
        H = -(p * p.log()).sum(dim=-1) / math.log(x.shape[-1] + self.eps)  # [B,L-1]

        # 3) Mutual-info proxy (I): Entropy based on pairwise similarity
        z = _normalize_l2(x, dim=-1, eps=self.eps)
        sim = torch.bmm(z, z.transpose(1, 2))     # [B,L-1,L-1]
        # Mask out self-similarity (diagonal)
        eye = torch.eye(sim.size(-1), device=sim.device).bool().unsqueeze(0)
        sim = sim.masked_fill(eye, -1e9)
        # Apply softmax for similarity distribution
        q = _safe_softmax(sim, dim=-1, tau=self.tau_sim, eps=self.eps)
        # Calculate similarity entropy (Hsim) and proxy I = 1 - Hsim
        Hsim = -(q * q.log()).sum(dim=-1) / math.log(q.shape[-1] + self.eps)  # [B,L-1]
        I = 1.0 - Hsim

        # 4) Z-score normalization + linear combination
        s1 = _zscore(s_attn, dim=1, eps=self.eps)
        s2 = _zscore(H, dim=1, eps=self.eps)
        s3 = _zscore(I, dim=1, eps=self.eps)
        a = self.alphas
        # Return combined hybrid score
        return a.attn * s1 + a.entropy * s2 + a.mutual * s3    # [B,L-1]

    def forward(
        self,
        layers: List[Dict[str, torch.Tensor]],
        cross_attn: Optional[torch.Tensor] = None,
        cross_last_dim_is_L: bool = True,
    ) -> torch.Tensor:
        # Calculate hybrid score for each layer
        per_layer = [self._single_layer_hybrid(d['attn'], d['keys'])
                     for d in layers]

        fused = per_layer[0]
        if len(per_layer) > 1:
            # Normalize scores from all layers
            normed = [_zscore(s, dim=1, eps=self.eps) for s in per_layer]
            # Calculate layer weights using softmax over gates
            if self.layer_gates is not None:
                g = torch.softmax(self.layer_gates, dim=0)
            else:
                # Use uniform weights if gates are not learnable/used
                g = torch.full(
                    (len(normed),),
                    1.0 / len(normed),
                    device=fused.device
                )
            # Weighted sum of normalized scores
            fused = sum(g[i] * normed[i] for i in range(len(normed)))

        # Optional cross-attention fusion
        if cross_attn is not None and self.cross_beta != 0.0:
            # Average cross-attention weights over all dimensions except Batch and Token/Sequence length
            if cross_last_dim_is_L:
                s_cross = cross_attn.mean(
                    dim=tuple(range(1, cross_attn.dim() - 1))
                )  # [B,L]
            else:
                s_cross = cross_attn.mean(dim=(1, 3))
            s_cross = s_cross[:, 1:]  # Remove CLS token score
            fused = fused + self.cross_beta * _zscore(
                s_cross, dim=1, eps=self.eps
            )
        return fused  # [B,L-1]

# --------- Dynamic K --------- #

@dataclass
class DynamicKConfig:
    # Configuration for determining K dynamically based on score variance
    c: float = 8.0          # Constant offset/bias
    k_min: int = 4          # Minimum K value
    k_max: int = 64         # Maximum K value
    eps: float = 1e-6

def dynamic_k_from_scores(
    scores: torch.Tensor,
    cfg: DynamicKConfig
) -> torch.Tensor:
    # scores: [B,L-1]
    # Calculate variance across patch scores for each image in the batch
    var = scores.var(dim=1, unbiased=False)  # [B]
    # Calculate K based on variance (log(var) + constant offset)
    k = torch.log(var + cfg.eps) + cfg.c     # [B]
    # Round, clamp to [k_min, k_max], and ensure K >= 1
    k = torch.round(k).clamp(
        min=cfg.k_min, max=cfg.k_max
    ).to(torch.int64)
    return torch.clamp(k, min=1)

# --------- Hierarchical merge --------- #

@dataclass
class MergingConfig:
    # Configuration for merging remaining tokens into contextual tokens
    contextual_num: int = 16
    kmeans_init_factor: float = 2.0
    kmeans_iters: int = 10
    agglomerative: bool = True # Use agglomerative merging after k-means initialization
    eps: float = 1e-12

def _kmeans_one(
    x: torch.Tensor,
    k: int,
    w: Optional[torch.Tensor],
    iters: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Performs weighted k-means clustering for a single sample (unbatched)
    device = x.device
    N, D = x.shape
    k = min(max(1, k), N)

    # k-means++ initialization
    idx0 = torch.randint(0, N, (1,), device=device)
    cents = [x[idx0]]
    if k > 1:
        d2 = torch.cdist(x, cents[0], p=2).squeeze(-1) ** 2
        for _ in range(1, k):
            probs = d2 / (d2.sum() + 1e-12)
            nxt = torch.multinomial(probs.clamp_min(1e-12), 1)
            cents.append(x[nxt])
            d2 = torch.minimum(
                d2,
                (torch.cdist(x, cents[-1], p=2).squeeze(-1) ** 2),
            )
    centroids = torch.cat(cents, dim=0)  # [k,D]

    assign = torch.zeros(N, dtype=torch.long, device=device)
    for _ in range(iters):
        # E-step: Assign points to nearest centroid
        dist = torch.cdist(x, centroids, p=2)  # [N,k]
        assign = dist.argmin(dim=1)            # [N]
        # M-step: Update centroids (weighted mean)
        for ci in range(centroids.shape[0]):
            mask = (assign == ci)
            if mask.any():
                if w is not None:
                    # Weighted mean calculation
                    ww = w[mask].unsqueeze(-1)     # [n_i,1]
                    centroids[ci] = (ww * x[mask]).sum(dim=0) / (
                        ww.sum() + 1e-12
                    )
                else:
                    # Simple mean calculation
                    centroids[ci] = x[mask].mean(dim=0)
            else:
                # Empty cluster -> random re-initialization
                ridx = torch.randint(0, N, (1,), device=device)
                centroids[ci] = x[ridx]
    return centroids, assign

def _agglomerative_merge(
    centroids: torch.Tensor,
    target_k: int
) -> torch.Tensor:
    # Agglomerative (hierarchical) merging based on similarity
    # centroids: [K,D]
    while centroids.shape[0] > target_k:
        z = _normalize_l2(centroids, dim=-1)
        # Calculate pairwise cosine similarity
        sim = torch.matmul(z, z.T)    # [K,K]
        sim.fill_diagonal_(-1.0)
        # Find the most similar pair (i,j)
        flat_idx = torch.argmax(sim).item()
        i = flat_idx // sim.shape[1]
        j = flat_idx % sim.shape[1]
        if i > j:
            i, j = j, i
        # Merge by simple averaging
        merged = (centroids[i] + centroids[j]) / 2.0
        # Reconstruct the centroid list, excluding i and j, adding the merged one
        centroids = torch.cat(
            [centroids[:i], centroids[i+1:j], centroids[j+1:], merged.unsqueeze(0)],
            dim=0,
        )
    return centroids

def hierarchical_context_merge(
    hidden_remain: torch.Tensor,   # [B,Nr,C] Remaining hidden states
    keys_remain: torch.Tensor,     # [B,Nr,Ck] Remaining scoring keys
    weights_remain: Optional[torch.Tensor],
    cfg: MergingConfig,
) -> torch.Tensor:
    # Merges remaining tokens into 'contextual_num' tokens using clustering
    B, Nr, C = hidden_remain.shape
    z_all = _normalize_l2(keys_remain.float(), dim=-1) # Normalized keys for clustering
    contextual_tokens = []

    for b in range(B):
        z = z_all[b]             # [Nr,Ck]
        h = hidden_remain[b]     # [Nr,C]
        w = None if weights_remain is None else weights_remain[b]  # [Nr]?

        # Determine K for initial k-means step
        init_k = min(
            int(max(cfg.contextual_num,
                    math.ceil(cfg.kmeans_init_factor * cfg.contextual_num))),
            Nr,
        )
        # 1. K-means clustering (weighted)
        cents, assign = _kmeans_one(z, init_k, w, iters=cfg.kmeans_iters)
        
        # 2. Agglomerative merging (optional)
        if cfg.agglomerative and init_k > cfg.contextual_num:
            cents = _agglomerative_merge(cents, cfg.contextual_num)

        # 3. Final assignment based on merged centroids
        dist = torch.cdist(z, cents, p=2)   # [Nr,ctx_num]
        final_ids = dist.argmin(dim=1)      # [Nr]

        # 4. Aggregate hidden states based on final assignment
        ctoks = []
        for ci in range(cfg.contextual_num):
            mask = (final_ids == ci)
            if not mask.any():
                # If cluster is empty, append zero vector
                ctoks.append(torch.zeros(C, device=h.device, dtype=h.dtype))
                continue
            if w is not None:
                # Weighted aggregation
                ww = w[mask].unsqueeze(-1)      # [n_i,1]
                agg = (ww * h[mask]).sum(dim=0) / (ww.sum() + cfg.eps)
            else:
                # Simple mean aggregation
                agg = h[mask].mean(dim=0)
            ctoks.append(agg)
        contextual_tokens.append(torch.stack(ctoks, dim=0))  # [ctx_num,C]

    return torch.stack(contextual_tokens, dim=0)   # [B,ctx_num,C]

# --------- compression wrapper --------- #

@dataclass
class CompressionConfig:
    # Overall configuration for the compression process
    alphas: ScoringAlphas = field(default_factory=ScoringAlphas)
    tau_feat: float = 0.2
    tau_sim: float = 0.1
    cross_beta: float = 0.0
    dynamic_k: bool = True
    dynk: DynamicKConfig = field(default_factory=DynamicKConfig)
    k_min: int = 4
    k_max: int = 64
    merging: MergingConfig = field(default_factory=MergingConfig)
    skip_ctx_merge: bool = True

class VisionZipCompressor(nn.Module):
    def __init__(
        self,
        num_scoring_layers: int = 1,
        cfg: Optional[CompressionConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else CompressionConfig()
        # Initialize scorer module
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
        scoring_layers: List[Dict[str, torch.Tensor]],  # Per layer: {attn:[B,H,L,L], keys:[B,L,Ck]}
        hidden_states: torch.Tensor,                    # [B,L,C] Full sequence hidden states
        cross_attn: Optional[torch.Tensor] = None,
        cross_last_dim_is_L: bool = True,
        dominant_num: Optional[int] = None,             # Explicit K override (disables dynamic-K)
        weights_for_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C = hidden_states.shape

        # 1) Multi-signal scoring
        scores = self.scorer(
            scoring_layers,
            cross_attn=cross_attn,
            cross_last_dim_is_L=cross_last_dim_is_L,
        )  # [B,L-1]

        # 2) Dynamic-K determination per image
        if self.cfg.dynamic_k and dominant_num is None:
            k_per = dynamic_k_from_scores(scores, self.cfg.dynk)  # [B]
        else:
            # Use fixed K (from dominant_num or default min K)
            k_val = dominant_num if dominant_num is not None else max(
                self.cfg.k_min, 4
            )
            k_per = torch.full(
                (B,),
                int(k_val),
                dtype=torch.long,
                device=hidden_states.device,
            )
        # Final clamp to ensure K stays within the allowed range
        k_per = torch.clamp(
            k_per,
            min=self.cfg.k_min,
            max=self.cfg.k_max,
        )

        outputs, idx_batches = [], []

        # 3) Per image processing: dominant selection + contextual merge
        for b in range(B):
            k = int(k_per[b].item())
            # Select top-k scores from the L-1 patch tokens
            topk = torch.topk(scores[b], k)       
            dom_idx = topk.indices + 1           # +1 to map back to original sequence index
            
            # Indices to keep (CLS token + K dominant patches)
            keep_idx = torch.cat(
                [
                    torch.tensor(
                        [0],
                        device=hidden_states.device,
                        dtype=dom_idx.dtype,
                    ),
                    dom_idx,
                ],
                dim=0,
            )                                    

            # Create mask for tokens to remain/be merged
            mask = torch.ones(L, dtype=torch.bool, device=hidden_states.device)
            mask[keep_idx] = False

            dominant_tokens = hidden_states[b, ~mask, :]   # [1+K,C] Tokens to keep (CLS + dominant)
            remain_hidden = hidden_states[b, mask, :]      # [Nr,C] Tokens remaining for merging
            remain_keys = scoring_layers[0]['keys'][b, mask, :]  # [Nr,Ck] Keys for remaining tokens

            # Handle optional weights for contextual tokens
            if weights_for_context is not None:
                w_all = weights_for_context[b]            # [L-1] (Patch weights, no CLS)
                abs_idx = torch.arange(L, device=hidden_states.device)
                rem_abs = abs_idx[mask]                   # indices of remaining tokens
                # w_all is patch-only weight, so index requires -1 offset
                w_rem = w_all[rem_abs - 1]
            else:
                w_rem = None

            # ====== Contextual merge ======
            ctx_num = self.cfg.merging.contextual_num
            Nr = remain_hidden.shape[0]

            if self.cfg.skip_ctx_merge:
                # ====== VisionZip-style contextual token aggregation ======
                if ctx_num <= 0:
                    ctx_tokens = remain_hidden.new_zeros((0, C))
                elif Nr == 0:
                    ctx_tokens = remain_hidden.new_zeros((ctx_num, C))
                else:
                    metric_filtered = remain_keys.unsqueeze(0).float()        # [1,Nr,Ck]
                    metric_normalized = _normalize_l2(metric_filtered, dim=-1)  # [1,Nr,Ck]

                    step = max(1, Nr // ctx_num)
                    target_indices = torch.arange(
                        0, Nr, step, device=remain_hidden.device
                    )[:ctx_num]   # [<=ctx_num]

                    if target_indices.numel() < ctx_num:
                        last_idx = target_indices[-1]
                        pad_idx = last_idx.repeat(ctx_num - target_indices.numel())
                        target_indices = torch.cat([target_indices, pad_idx], dim=0)

                    target_tokens = metric_normalized[:, target_indices, :]    # [1,ctx_num,Ck]

                    all_nd_idx = torch.arange(Nr, device=remain_hidden.device)
                    remain_mask2 = ~torch.isin(all_nd_idx, target_indices)
                    tokens_to_merge = metric_normalized[:, remain_mask2, :]    # [1,Nr_rem,Ck]

                    if tokens_to_merge.shape[1] == 0:
                        ctx_tokens = remain_hidden[target_indices, :]          # [ctx_num,C]
                    else:
                        similarity = torch.bmm(
                            tokens_to_merge,
                            target_tokens.transpose(1, 2)
                        )  # [1,Nr_rem,ctx_num]

                        assign_one_hot = torch.zeros(
                            1,
                            tokens_to_merge.shape[1],
                            ctx_num,
                            dtype=remain_hidden.dtype,
                            device=remain_hidden.device,
                        )
                        assign_one_hot.scatter_(
                            2,
                            similarity.argmax(dim=2, keepdim=True),
                            1,
                        )
                        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [1,ctx_num,1]

                        hidden_to_merge = remain_hidden[remain_mask2, :].unsqueeze(0)  # [1,Nr_rem,C]
                        aggregated_hidden = torch.bmm(
                            assign_one_hot.transpose(1, 2),
                            hidden_to_merge,
                        ) / counts                                                      # [1,ctx_num,C]

                        target_hidden = remain_hidden[target_indices, :].unsqueeze(0)   # [1,ctx_num,C]

                        # 6) VisionZip: target_hidden + aggregated_hidden
                        ctx_tokens = (target_hidden + aggregated_hidden)[0]             # [ctx_num,C]

            else:
                # ====== SparseZip kmeans + agglomerative merge ======
                ctx_tokens = hierarchical_context_merge(
                    remain_hidden.unsqueeze(0),
                    remain_keys.unsqueeze(0),
                    None if w_rem is None else w_rem.unsqueeze(0),
                    cfg=self.cfg.merging,
                )[0]    # [contextual_num, C]

            out = torch.cat([dominant_tokens, ctx_tokens], dim=0)  # [1+K+ctx_num, C]
            outputs.append(out)
            idx_batches.append(keep_idx)

        # 4) Pad outputs to the maximum length within the batch
        max_len = max(o.shape[0] for o in outputs)
        padded = []
        for o in outputs:
            if o.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - o.shape[0],
                    C,
                    device=o.device,
                    dtype=o.dtype,
                )
                o = torch.cat([o, pad], dim=0)
            padded.append(o)
        out_tokens = torch.stack(padded, dim=0)   # [B,max_len,C]

        # Also pad indices, using -1 for empty slots
        max_idx_len = max(len(ids) for ids in idx_batches)
        idx_padded = []
        for ids in idx_batches:
            if len(ids) < max_idx_len:
                pad = -torch.ones(
                    max_idx_len - len(ids),
                    device=ids.device,
                    dtype=ids.dtype,
                )
                ids = torch.cat([ids, pad], dim=0)
            idx_padded.append(ids)
        all_indices = torch.stack(idx_padded, dim=0)  # [B,max_idx_len]

        return out_tokens, all_indices