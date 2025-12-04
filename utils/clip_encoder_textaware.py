# # utils/clip_encoder_textaware.py
import torch
from utils.visionzip_textaware import TextContext

@torch.no_grad()
def clipvisiontower_visionzip_textaware_forward(self, images):
    """
    This forward is intended for CLIPVisionTower_VisionZip:
    - self: CLIPVisionTower_VisionZip instance
    - images: torch.Tensor or list[Tensor]
    """
    # 1) List case: Keep original logic (no compression)
    if isinstance(images, list):
        image_features = []
        for image in images:
            image_forward_out = self.vision_tower(
                image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                output_hidden_states=True,
                output_attentions=True,
            )
            image_feature = self.feature_select(image_forward_out).to(image.dtype)
            image_features.append(image_feature)
        return image_features

    # 2) Batched case: Take the second to last layer, same as original VisionZip
    image_forward_outs = self.vision_tower(
        images.to(device=self.device, dtype=self.dtype),
        output_hidden_states=True,
        output_attentions=True,
    )
    attn_weights  = image_forward_outs.attentions[-2]    # [B, H, L, L]
    hidden_states = image_forward_outs.hidden_states[-2] # [B, L, C_vis]

    metric = self.vision_tower.vision_model.encoder.layers[-2].metric
    dominant_num   = self.vision_tower._info["dominant"]
    contextual_num = self.vision_tower._info["contextual"]

    B, L, C_vis = hidden_states.shape

    # ---- Original CLS Score ----
    cls_idx = 0
    cls_attention     = attn_weights[:, :, cls_idx, cls_idx + 1 :]  # [B, H, L-1]
    cls_attention_sum = cls_attention.sum(dim=1)                    # [B, L-1]
    final_score = cls_attention_sum

    # ---- Text-aware Score (entirely in LLM space) ----
    text_ctx = TextContext.get_text_query()              # [B, 1, D_llm]
    mm_proj  = getattr(self, "mm_projector", None)

    if (text_ctx is not None) and (mm_proj is not None):
        # 1) Patch tokens in vision space
        patch_tokens = hidden_states[:, 1:, :]           # [B, L-1, C_vis]

        # 2) Project to LLM space: [B, L-1, D_llm]
        patch_tokens_llm = mm_proj(patch_tokens)

        # 3) text_ctx is already a query in LLM space: [B, 1, D_llm]
        text_ctx = text_ctx.to(dtype=patch_tokens_llm.dtype,
                               device=patch_tokens_llm.device)

        # 4) Dot product or cosine similarity
        text_score = torch.bmm(
            patch_tokens_llm,
            text_ctx.transpose(1, 2)
        ).squeeze(-1)                                    # [B, L-1]

        # 5) Normalize + Fuse
        v_min, v_max = cls_attention_sum.min(dim=1, keepdim=True)[0], cls_attention_sum.max(dim=1, keepdim=True)[0]
        cls_norm  = (cls_attention_sum - v_min) / (v_max - v_min + 1e-6)

        t_min, t_max = text_score.min(dim=1, keepdim=True)[0], text_score.max(dim=1, keepdim=True)[0]
        text_norm = (text_score - t_min) / (t_max - t_min + 1e-6)

        alpha = float(getattr(self, "vis_alpha", 0.8))
        final_score = alpha * cls_norm + (1.0 - alpha) * text_norm

    
    # --- Dominant selection: Use final_score top-k ---
    topk_indices = final_score.topk(dominant_num, dim=1).indices + 1  # [B, K]
    all_indices = torch.cat(
        [
            torch.zeros(
                (hidden_states.shape[0], 1),
                dtype=topk_indices.dtype,
                device=topk_indices.device,
            ),
            topk_indices,
        ],
        dim=1,
    )  # [B, 1+K]

    mask = torch.ones_like(
        hidden_states[:, :, 0],
        dtype=torch.bool,
        device=metric.device,
    ).scatter_(1, all_indices, False)

    dominant_tokens = (
        hidden_states
        .masked_select(~mask.unsqueeze(-1))
        .view(hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2])
    )  # [B, 1+K, C]

    # --- Contextual: Follows VisionZip implementation exactly ---
    metric_filtered = metric[mask].view(
        hidden_states.shape[0],
        hidden_states.shape[1] - (dominant_num + 1),
        metric.shape[2],
    )

    hidden_states_filtered = (
        hidden_states
        .masked_select(mask.unsqueeze(-1))
        .view(
            hidden_states.shape[0],
            hidden_states.shape[1] - (dominant_num + 1),
            hidden_states.shape[2],
        )
    )

    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)

    step = max(1, metric_normalized.shape[1] // contextual_num)
    target_indices = torch.arange(
        0,
        metric_normalized.shape[1],
        step,
        device=metric_normalized.device,
    )[:contextual_num]
    target_tokens = metric_normalized[:, target_indices, :]

    all_idx = torch.arange(metric_normalized.shape[1], device=metric_normalized.device)
    is_target = torch.isin(all_idx, target_indices)

    tokens_to_merge = metric_normalized[:, ~is_target, :]
    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))

    assign_one_hot = torch.zeros(
        tokens_to_merge.shape[0],
        tokens_to_merge.shape[1],
        contextual_num,
        dtype=hidden_states_filtered.dtype,
        device=metric_normalized.device,
    )
    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)

    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)

    hidden_to_merge = hidden_states_filtered[:, ~is_target, :]
    aggregated_hidden = torch.bmm(
        assign_one_hot.transpose(1, 2),
        hidden_to_merge,
    ) / counts

    target_hidden = hidden_states_filtered[:, target_indices, :]
    contextual_tokens = target_hidden + aggregated_hidden

    hidden_states_save = torch.cat(
        [dominant_tokens, contextual_tokens],
        dim=1,
    ).to(images.dtype)

    return hidden_states_save, all_indices