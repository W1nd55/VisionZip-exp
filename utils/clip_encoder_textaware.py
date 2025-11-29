# utils/clip_encoder_textaware.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _zscore(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    m = x.mean(dim=dim, keepdim=True)
    s = x.std(dim=dim, keepdim=True)
    return (x - m) / (s + eps)

# utils/clip_encoder_textaware.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zscore(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    m = x.mean(dim=dim, keepdim=True)
    s = x.std(dim=dim, keepdim=True)
    return (x - m) / (s + eps)


class CLIPVisionTower_VisionZip_TextAware(nn.Module):

    @torch.no_grad()
    def forward(self, images, *args, **kwargs):
        """
        images: Tensor [B, C, H, W] 或 list[Tensor]
            hidden_states_save: [B, 1 + K_dom + K_ctx, D]
        """

        if isinstance(images, list):
            image_features = []
            for image in images:
                outs = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    output_attentions=True,
                )
                feat = self.feature_select(outs).to(image.dtype)
                image_features.append(feat)
            return torch.stack(image_features, dim=0)

        outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=True,
        )

        attn_weights = outs.attentions[-2]       # [B, H, T, T]
        hidden_states = outs.hidden_states[-2]   # [B, T, D]
        metric = self.vision_tower.vision_model.encoder.layers[-2].metric  # [B, T, Ck]

        B, L_raw, D = hidden_states.shape
        L_patches = L_raw - 1
        device = hidden_states.device

        cls_idx = 0
        cls_attention = attn_weights[:, :, cls_idx, cls_idx + 1 :]   # [B, H, L_patches]
        Sd = cls_attention.sum(dim=1)                                # [B, L_patches]

        patch_keys = metric[:, 1:, :]   # [B, L_patches, Ck]
        Ck = patch_keys.shape[-1]

        text_emb = getattr(self, "visionzip_text_embed", None)
        alpha = float(getattr(self, "visionzip_alpha", 0.5))

        if text_emb is not None:
            # 支持 [Ck] 或 [B, Ck]
            if text_emb.dim() == 1:
                text_emb = text_emb.unsqueeze(0).expand(B, -1)  # [B, Ck]
            elif text_emb.dim() == 2 and text_emb.shape[0] == 1 and B > 1:
                text_emb = text_emb.expand(B, -1)               # [B, Ck]
            elif text_emb.dim() != 2 or text_emb.shape[-1] != Ck:
                raise ValueError(
                    f"visionzip_text_embed shape mismatch: expected [..., {Ck}], got {tuple(text_emb.shape)}"
                )

            text_emb = text_emb.to(device=patch_keys.device, dtype=patch_keys.dtype)  # [B, Ck]

            text_norm = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-12)         # [B, Ck]
            patch_norm = patch_keys / (patch_keys.norm(dim=-1, keepdim=True) + 1e-12)    # [B, L_patches, Ck]

            cos_score = (patch_norm * text_norm.unsqueeze(1)).sum(dim=-1)                # [B, L_patches]

            Sd_n = _zscore(Sd, dim=-1)
            cos_n = _zscore(cos_score, dim=-1)
            score = alpha * Sd_n + (1.0 - alpha) * cos_n                                  # [B, L_patches]
        else:
            score = _zscore(Sd, dim=-1)                                                  # [B, L_patches]

        dominant_num = int(getattr(self, "visionzip_dominant", 54))
        contextual_num = int(getattr(self, "visionzip_contextual", 10))

        dominant_num = max(1, min(dominant_num, L_patches))

        topk_indices = score.topk(dominant_num, dim=1).indices + 1  

        cls_indices = torch.zeros((B, 1), dtype=topk_indices.dtype, device=device)  # CLS 在 index 0

        all_indices = torch.cat([cls_indices, topk_indices], dim=1)

        mask = torch.ones((B, L_raw), dtype=torch.bool, device=device)
        mask.scatter_(1, all_indices, False)

        # dominant_tokens: [B, 1 + K_dom, D]
        dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(
            B, dominant_num + 1, D
        )

        Nr = L_patches - dominant_num
        contextual_num = max(0, min(contextual_num, Nr))

        if Nr > 0 and contextual_num > 0:
            metric_filtered = metric.masked_select(mask.unsqueeze(-1)).view(
                B, Nr, metric.shape[2]
            )  # [B, Nr, Ck]
            hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(
                B, Nr, D
            )  # [B, Nr, D]

            metric_normalized = metric_filtered / (
                metric_filtered.norm(dim=-1, keepdim=True) + 1e-12
            )  # [B, Nr, Ck]

            step = max(1, Nr // contextual_num)
            target_indices = torch.arange(0, Nr, step, device=device)[:contextual_num]

            if target_indices.numel() < contextual_num:
                last_idx = target_indices[-1] if target_indices.numel() > 0 else 0
                pad_idx = last_idx.repeat(contextual_num - target_indices.numel())
                target_indices = torch.cat([target_indices, pad_idx], dim=0)

            target_tokens = metric_normalized[:, target_indices, :]  # [B, K_ctx, Ck]

            all_nd_idx = torch.arange(Nr, device=device)
            remain_mask = ~torch.isin(all_nd_idx, target_indices)

            tokens_to_merge = metric_normalized[:, remain_mask, :]      # [B, Nr-K_ctx, Ck]
            similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))  # [B, Nr-K_ctx, K_ctx]

            assign_one_hot = torch.zeros(
                B,
                tokens_to_merge.shape[1],
                contextual_num,
                dtype=hidden_states_filtered.dtype,
                device=device,
            )
            assign_one_hot.scatter_(
                2, similarity.argmax(dim=2).unsqueeze(-1), 1
            )  # [B, Nr-K_ctx, K_ctx]
            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)      # [B, K_ctx, 1]

            hidden_to_merge = hidden_states_filtered[:, remain_mask, :]        # [B, Nr-K_ctx, D]
            aggregated_hidden = torch.bmm(
                assign_one_hot.transpose(1, 2), hidden_to_merge
            ) / counts                                                         # [B, K_ctx, D]

            target_hidden = hidden_states_filtered[:, target_indices, :]       # [B, K_ctx, D]

            contextual_tokens = target_hidden + aggregated_hidden              # [B, K_ctx, D]
        else:
            contextual_tokens = hidden_states[:, 0:0, :]  # [B, 0, D] 

        hidden_states_save = torch.cat(
            [dominant_tokens, contextual_tokens], dim=1
        ).to(images.dtype)  # [B, 1+K_dom+K_ctx, D]

        self.visionzip_last_indices = all_indices  # [B, 1+K_dom]

        return hidden_states_save