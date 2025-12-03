# utils/clip_encoder_sparsezip_exp.py
import torch
import torch.nn as nn
from typing import Optional, List, Dict

# class CLIPVisionTower_SparseZip_EXP(nn.Module):

#     @torch.no_grad()
#     def forward(self, images):
#         """
#         self: This is actually an instance of llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower
#               It should already contain:
#                 - self.vision_tower: Hugging Face's CLIPVisionModel
#                 - self.device / self.dtype
#                 - self._info["dominant"], self._info["contextual"]
#                 - (Attached during LlavaSparseZipModel construction) self._vz_comp: VisionZipCompressor
#         """

#         # 1) List input: Typically used for multi-image modes like chat/serve. 
#         #    Maintain original feature_select logic; SparseZip is temporarily skipped.
#         if isinstance(images, list):
#             image_features = []
#             for image in images:
#                 outs = self.vision_tower(
#                     image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
#                     output_hidden_states=True,
#                     output_attentions=True,
#                 )
#                 feat = self.feature_select(outs).to(image.dtype)
#                 image_features.append(feat)
#             return torch.stack(image_features, dim=0), None

#         # 2) Batch input: The path for actual SparseZip compression
#         # images: [B, 3, H, W]
#         outs = self.vision_tower(
#             images.to(device=self.device, dtype=self.dtype),
#             output_hidden_states=True,
#             output_attentions=True,
#         )
#         attn_weights  = outs.attentions[-2]       # [B, H, L, L]
#         hidden_states = outs.hidden_states[-2]    # [B, L, C]
#         # The metric is the key head-mean stored via monkey-patching in CLIPEncoderLayer.forward
#         metric = self.vision_tower.vision_model.encoder.layers[-2].metric  # [B, L, Ck]

#         # --- Prepare inputs for the compressor ---
#         scoring_layers = [{
#             "attn": attn_weights,   # [B, H, L, L]
#             "keys": metric,         # [B, L, Ck]
#         }]

#         # Optional cross-attention, none here for now
#         cross_attn = None

#         # 3) Retrieve the compressor attached to the vision tower
#         if not hasattr(self, "_vz_comp") or self._vz_comp is None:
#             raise RuntimeError(
#                 "[SparseZip] _vz_comp not found on CLIPVisionTower. "
#                 "Make sure you build the model via LlavaSparseZipModel "
#                 "so that vt._vz_comp is attached."
#             )

#         # Strategy for selecting dominant K:
#         #   - If you want to "fully rely on dynamic-K": dominant_num = None
#         #   - If you want to fix it to the dominant value in cfg: dominant_num = int(self._info['dominant'])
#         # Since VisionZipCompressor already includes dynamic_k logic, None is recommended here to let it decide.
#         dominant_num = None    # or int(self._info["dominant"])

#         # 4) Call the SparseZip compressor: scores -> K dominant + hierarchical contextual merge
#         out_tokens, all_indices = self._vz_comp(
#             scoring_layers=scoring_layers,
#             hidden_states=hidden_states,          # [B, L, C]
#             cross_attn=cross_attn,
#             cross_last_dim_is_L=True,
#             dominant_num=dominant_num,
#             weights_for_context=None,            # Additional weights can be added here if needed later
#         )
#         # out_tokens: [B, K+1+ctx, C]
#         # all_indices: [B, <=L], indices of original tokens (CLS + dominant; others only participate in aggregation during merge)

#         out_tokens = out_tokens.to(images.dtype)
#         return out_tokens, all_indices

class CLIPVisionTower_SparseZip_EXP(nn.Module):

    @torch.no_grad()
    def forward(self, images, text_embeds=None):
        """
        Args:
            images: [B, 3, H, W] input images
            text_embeds: [B, T, D] text query embeddings (optional, for cross-attention)
        """
        # If text_embeds not provided, check if stored in instance variable
        if text_embeds is None and hasattr(self, "_text_embeds"):
            text_embeds = self._text_embeds
        
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

            feats = torch.stack(image_features, dim=0)  # [B, T, C]
            self._last_keep_idx = None
            return feats, None

        outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=True,
        )
        attn_weights  = outs.attentions[-2]       # [B, H, L, L]
        hidden_states = outs.hidden_states[-2]    # [B, L, C]
        metric = self.vision_tower.vision_model.encoder.layers[-2].metric  # [B, L, Ck]

        scoring_layers = [{
            "attn": attn_weights,   # [B, H, L, L]
            "keys": metric,         # [B, L, Ck]
        }]
        
        # Compute cross-attention if text embeddings are provided
        cross_attn = None
        if text_embeds is not None and hasattr(self, "_vz_comp") and self._vz_comp.cfg.cross_beta > 0.0:
            # text_embeds: [B, T, D]
            # hidden_states: [B, L, C]
            # We compute Q_text @ K_image^T to get cross-attention scores
            
            # Project text to same dim as image keys if needed
            B, L, C = hidden_states.shape
            T = text_embeds.shape[1]
            
            # Use keys from vision encoder for cross-attention computation
            # Keys: [B, L, Ck], Text: [B, T, D]
            # For simplicity, use mean-pooled text embedding as query
            text_query = text_embeds.mean(dim=1, keepdim=True)  # [B, 1, D]
            
            # Compute similarity (simplified cross-attention)
            # Normalize and compute dot product
            text_norm = torch.nn.functional.normalize(text_query.float(), dim=-1)  # [B, 1, D]
            keys_norm = torch.nn.functional.normalize(metric.float(), dim=-1)  # [B, L, Ck]
            
            # If dimensions don't match, project text to key dimension
            if text_query.shape[-1] != metric.shape[-1]:
                # Simple linear projection (we'll use a learnable projection later if needed)
                # For now, use averaging to match dimensions
                if text_query.shape[-1] > metric.shape[-1]:
                    # Downsample text features
                    text_norm = text_norm[..., :metric.shape[-1]]
                else:
                    # Pad text features
                    pad_size = metric.shape[-1] - text_query.shape[-1]
                    text_norm = torch.nn.functional.pad(text_norm, (0, pad_size))
            
            # Compute cross-attention: [B, 1, Ck] @ [B, Ck, L] -> [B, 1, L]
            cross_attn = torch.bmm(text_norm, keys_norm.transpose(1, 2))  # [B, 1, L]
            
            # Expand to match expected format [B, H, 1, L] for compatibility
            # (The compressor will average over H and the query dim anyway)
            cross_attn = cross_attn.unsqueeze(1)  # [B, 1, 1, L]

        if not hasattr(self, "_vz_comp") or self._vz_comp is None:
            raise RuntimeError(
                "[SparseZip] _vz_comp not found on CLIPVisionTower. "
                "Make sure you build the model via LlavaSparseZipModel "
                "so that vt._vz_comp is attached."
            )

        dominant_num = None

        out_tokens, all_indices = self._vz_comp(
            scoring_layers=scoring_layers,
            hidden_states=hidden_states,          # [B, L, C]
            cross_attn=cross_attn,
            cross_last_dim_is_L=True,
            dominant_num=dominant_num,
            weights_for_context=None,
        )
        # out_tokens: [B, K+1+ctx, C]
        # all_indices: [B, <=L]

        out_tokens = out_tokens.to(images.dtype)

        self._last_keep_idx = all_indices

        return out_tokens, all_indices