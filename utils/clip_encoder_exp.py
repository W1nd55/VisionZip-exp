import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def hybrid_token_score(attn_weights, hidden_states, metric,
                       alpha=(1.0, 0.4, 0.6),
                       tau_feat=0.2, tau_sim=0.1, eps=1e-12):
    """
    Computes a hybrid importance score for vision tokens by combining three factors:
    1. Attention from the CLS token (s_attn).
    2. Internal representation uncertainty (H_entropy).
    3. Mutual Information approximation with other tokens (I_mutual).

    Args:
      attn_weights: [B, H, L, L]  # Attention weights (e.g., from the last-2 layer)
      hidden_states: [B, L, C]    # Hidden states from the same layer
      metric: [B, L, Ck]          # Saved keys' head-mean representation (e.g., from monkey-patching)

    Returns:
      scores: [B, L-1]  # Mixed importance scores for each patch (excluding the CLS token)
    """

    B, H, L, _ = attn_weights.shape
    device = attn_weights.device

    # 1) s_attn: Score based on CLS -> patch attention
    cls_attn = attn_weights[:, :, 0, 1:]     # [B, H, L-1] - Extract CLS attention to all patch tokens
    s_attn = cls_attn.mean(dim=1)            # [B, L-1] - Average across heads (can also use .sum(dim=1))

    # 2) H_entropy: Internal representation uncertainty (Entropy after softmax across channels)
    #    Using 'metric' (key vectors) better aligns with the attention space;
    #    Temperature tau_feat controls the smoothness of the probability distribution.
    x = metric[:, 1:, :].float()             # [B, L-1, Ck] - Exclude CLS token
    p = F.softmax(x / max(tau_feat, eps), dim=-1).clamp_min(eps)
    H = -(p * (p.log())).sum(dim=-1)         # [B, L-1] - Calculate entropy
    # Normalize to [0,1] (Divide by the maximum possible entropy log(Ck))
    H = H / (torch.log(torch.tensor(x.shape[-1], device=device)) + eps)

    # 3) I_mutual: Mutual Information approximation with "other tokens" -- using the inverse of normalized similarity distribution entropy
    #    First, compute cosine similarity, then calculate the entropy of the similarity distribution for each token; I = 1 - H_norm
    z = F.normalize(metric[:, 1:, :].float(), dim=-1)     # [B, L-1, Ck] - L2-normalize key vectors
    sim = torch.bmm(z, z.transpose(1, 2))                 # [B, L-1, L-1] - Pairwise similarity matrix
    # Remove the influence of self-similarity (set diagonal to a large negative number)
    eye = torch.eye(sim.size(-1), device=device).unsqueeze(0)
    sim = sim.masked_fill(eye.bool(), -1e9)
    q = F.softmax(sim / max(tau_sim, eps), dim=-1).clamp_min(eps)
    Hsim = -(q * (q.log())).sum(dim=-1)                   # [B, L-1] - Calculate similarity entropy
    # Normalize similarity entropy
    Hsim = Hsim / (torch.log(torch.tensor(q.shape[-1], device=device)) + eps)
    I = 1.0 - Hsim                                        # The more "concentrated" (higher similarity to few others), the larger I (Mutual Information proxy)

    # 4) Robust normalization for different magnitudes (z-score or min-max can be used; z-score used here)
    def zscore(t):
        m = t.mean(dim=1, keepdim=True)
        s = t.std(dim=1, keepdim=True) + eps
        return (t - m) / s

    s1 = zscore(s_attn)
    s2 = zscore(H)
    s3 = zscore(I)

    a1, a2, a3 = alpha
    scores = a1 * s1 + a2 * s2 + a3 * s3                   # [B, L-1] - Final hybrid score
    return scores


class CLIPVisionTower_VisionZip_EXP(nn.Module):

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                outs = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True, output_attentions=True
                )
                feat = self.feature_select(outs).to(image.dtype)
                image_features.append(feat)
            return torch.stack(image_features, dim=0), None

        # —— not list：Dominant + Contextual —— 
        outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True, output_attentions=True
        )
        attn_weights  = outs.attentions[-2]
        hidden_states = outs.hidden_states[-2]
        metric = self.vision_tower.vision_model.encoder.layers[-2].metric
        dominant_num  = self.vision_tower._info["dominant"]
        contextual_num = self.vision_tower._info["contextual"]
        # —— Dominant Visual Tokens —— 
            # ----------------------------------
            # original dominent token selection based on cls attention
            # cls_idx = 0
            # cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]  
            # cls_attention_sum = cls_attention.sum(dim=1)  
            # topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
            # ----------------------------------
            ## New dominant token selection based on new scroe metric
        # —— hybrid scoring —— 
        use_hybrid = True
        scores = (hybrid_token_score(attn_weights, hidden_states, metric,
                                    alpha=(1.0, 0.4, 0.6),
                                    tau_feat=0.2, tau_sim=0.1) if use_hybrid else attn_weights[:, :, 0, 1:].mean(dim=1))
        topk_indices = scores.topk(dominant_num, dim=1).indices + 1

        all_indices = torch.cat(
            [torch.zeros((hidden_states.shape[0], 1), dtype=topk_indices.dtype, device=topk_indices.device),
             topk_indices], dim=1
        )
        mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)
        dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2])

        metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num + 1), metric.shape[2])
        hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num + 1), hidden_states.shape[2])

        metric_normalized = metric_filtered / (metric_filtered.norm(dim=-1, keepdim=True) + 1e-12)

        step = max(1, metric_normalized.shape[1] // contextual_num)
        target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
        target_tokens = metric_normalized[:, target_indices, :]

        all_nd_idx = torch.arange(metric_normalized.shape[1], device=metric_normalized.device)
        remain_mask = ~torch.isin(all_nd_idx, target_indices)

        tokens_to_merge = metric_normalized[:, remain_mask, :]
        similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))

        assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num,
                                     dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)

        hidden_to_merge = hidden_states_filtered[:, remain_mask, :]
        aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
        target_hidden = hidden_states_filtered[:, target_indices, :]

        contextual_tokens = target_hidden + aggregated_hidden
        hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(images.dtype)

        return hidden_states_save, all_indices






