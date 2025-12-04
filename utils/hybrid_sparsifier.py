# utils/hybrid_sparsifier.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HybridSparsifierConfig:
    prune_layers: Tuple[int, ...] = (2, 4, 6)
    alpha: float = 0.7
    beta: float = 2.0
    n_min: int = 24
    max_rank_k: int = 16
    n_text_raters: int = 8


class HybridSparsifier(nn.Module):
    def __init__(self, cfg: HybridSparsifierConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, T, D]
        attn_logits: torch.Tensor,        # [B, H, T, T] or [B, T, T]
        token_type_mask: torch.Tensor,    # [B, T]
        layer_idx: int,
    ): 
        cfg = self.cfg
        B, T, D = hidden_states.shape
        info = {}

        # not in prune list -> return directly
        if layer_idx not in cfg.prune_layers:
            return hidden_states, token_type_mask, None, info

        device = hidden_states.device

        new_h_list = []
        new_mask_list = []
        rank_list = []
        n_vis_list = []
        n_keep_list = []
        
        keep_indices_list = None 

        # [B, H, T, T] / [B, T, T]
        if attn_logits.ndim == 4:
            attn_mean = attn_logits.mean(dim=1)
        else:
            attn_mean = attn_logits

        for b in range(B):
            hs_b = hidden_states[b]
            mask_b = token_type_mask[b].bool()
            attn_b = attn_mean[b]

            vis_idx = torch.nonzero(mask_b, as_tuple=False).squeeze(-1)
            txt_idx = torch.nonzero(~mask_b, as_tuple=False).squeeze(-1)
            N_vis = vis_idx.numel()

            # --- Situation 1: little visual Token，no prune ---
            if N_vis <= cfg.n_min:
                new_h_list.append(hs_b)
                new_mask_list.append(token_type_mask[b])
                rank_list.append(torch.tensor(0, device=device))
                n_vis_list.append(torch.tensor(N_vis, device=device))
                n_keep_list.append(torch.tensor(N_vis, device=device))
                
                keep_indices_list = None 
                continue

            # --- Situation 2: normal prune ---
            
            # 1) text-aware score
            A = attn_b.softmax(dim=-1)
            if txt_idx.numel() == 0:
                text_score = torch.zeros(N_vis, device=device)
            else:
                A_tv = A[txt_idx][:, vis_idx]
                text_score_full = A_tv.sum(dim=1)
                R = min(cfg.n_text_raters, text_score_full.numel())
                top_r = text_score_full.topk(R).indices
                A_rv = A_tv[top_r]
                text_score = A_rv.mean(dim=0)
                text_score = text_score / (text_score.sum() + 1e-6)

            # 2) info-score
            H_v = hs_b[vis_idx]
            H_v_f32 = H_v.float()
            U, S, Vh = torch.linalg.svd(H_v_f32, full_matrices=False)
            r0 = min(cfg.max_rank_k, S.shape[0])
            U_r = U[:, :r0]
            S_r = S[:r0]
            info_score = (U_r ** 2) @ (S_r ** 2)
            info_score = info_score / (info_score.sum() + 1e-6)

            # 3) hybrid score & N_keep
            alpha = cfg.alpha
            hybrid_score = alpha * text_score + (1.0 - alpha) * info_score

            r = torch.tensor(0, device=device) 
            N_keep = min(cfg.n_min, N_vis)

            keep_local = hybrid_score.topk(N_keep).indices
            all_idx_local = torch.arange(N_vis, device=device)
            drop_mask_local = torch.ones(N_vis, dtype=torch.bool, device=device)
            drop_mask_local[keep_local] = False
            drop_local = all_idx_local[drop_mask_local]
            
            drop_idx = vis_idx[drop_local]

            # 4) [Disabled Merge] Just Prune
            keep_mask = torch.ones(T, dtype=torch.bool, device=device)
            keep_mask[drop_idx] = False
            
            current_keep_indices = torch.nonzero(keep_mask, as_tuple=True)[0]
            
            new_h = hs_b[keep_mask]
            new_m = token_type_mask[b][keep_mask]

            new_h_list.append(new_h)
            new_mask_list.append(new_m)
            
            keep_indices_list = current_keep_indices

            rank_list.append(r.to(device))
            n_vis_list.append(torch.tensor(N_vis, device=device))
            n_keep_list.append(torch.tensor(N_keep, device=device))

        if B != 1:
            # To be simple，Eval only support B=1
            # raise NotImplementedError("Eval stage pls use batch_size=1")
            pass

        new_hidden_states = new_h_list[0].unsqueeze(0)
        new_token_type_mask = new_mask_list[0].unsqueeze(0)

        info = {
            "layer_idx": layer_idx,
            "rank": int(rank_list[0].item()) if rank_list else 0,
            "n_vis": int(n_vis_list[0].item()) if n_vis_list else 0,
            "n_keep": int(n_keep_list[0].item()) if n_keep_list else 0,
        }
        
        return new_hidden_states, new_token_type_mask, keep_indices_list, info