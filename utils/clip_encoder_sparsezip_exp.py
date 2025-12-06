# utils/clip_encoder_sparsezip_exp.py
import torch
import torch.nn as nn
from typing import Optional, List, Dict
import torch.nn.functional as F
import math

class CLIPVisionTower_SparseZip_EXP(nn.Module):

    @torch.no_grad()
    def forward(self, images):
        """
        self: actually llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower 
                - self.vision_tower: HF CLIPVisionModel
                - self.device / self.dtype
                - self._info["dominant"], self._info["contextual"]
                - self._vz_comp: VisionZipCompressor
        """
        # 1) List input: multi-image 
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
            return torch.stack(image_features, dim=0), None

        # 2) Batch input
        # images: [B, 3, H, W]
        outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=True,
        )
        attn_weights  = outs.attentions[-2]       # [B, H, L, L]
        hidden_states = outs.hidden_states[-2]    # [B, L, C]
        # metric:  CLIPEncoderLayer.forward key，size [B, L, Ck]
        metric = self.vision_tower.vision_model.encoder.layers[-2].metric

        # --- SparseZip scoring input ---
        scoring_layers = [{
            "attn": attn_weights,
            "keys": metric,
        }]

        # ---------- Text-aware cross_attn ----------
        import math
        import torch.nn.functional as F

        cross_attn = None

        text_query   = getattr(self, "_text_query", None)          # [B, d_llm]
        inst_t2k     = getattr(self, "_text2key", None)          
        class_t2k    = getattr(type(self), "_text2key", None)    
        # print("[SparseZip Debug] self in CLIPVT forward:", type(self), "id=", id(self))

        if text_query is not None and (inst_t2k is not None or class_t2k is not None):
            try:
                # print("[SparseZip TextAware] CLIPVT forward: _text_query found, building cross_attn...")

                proj = inst_t2k if inst_t2k is not None else class_t2k

                metric_dtype  = metric.dtype
                metric_device = metric.device

                proj = proj.to(device=metric_device, dtype=metric_dtype)

                # q: [B, Ck] -> [B, 1, Ck]
                q = proj(text_query.to(device=metric_device, dtype=metric_dtype))  # [B, Ck]
                q = q.unsqueeze(1)                                                 # [B, 1, Ck]

                # k: [B, L, Ck]
                k = metric.to(device=metric_device, dtype=metric_dtype)           # [B, L, Ck]

                sim = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.size(-1)) # [B,1,L]
                cross_attn = F.softmax(sim, dim=-1)                               # [B,1,L]

                # print("[SparseZip TextAware] cross_attn shape:", cross_attn.shape,
                #       "dtype:", cross_attn.dtype)
            except Exception as e:
                print("[SparseZip TextAware] building cross_attn failed:", repr(e))
                cross_attn = None
        else:
            print(
                "[SparseZip TextAware] CLIPVT forward: no _text_query or _text2key/class_text2key, "
                "using vision-only scoring."
            )
        # -------------------------------------------

        # 3)  compressor
        if not hasattr(self, "_vz_comp") or self._vz_comp is None:
            raise RuntimeError(
                "[SparseZip] _vz_comp not found on CLIPVisionTower. "
                "Make sure you build the model via LlavaSparseZipModel so that vt._vz_comp is attached."
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

        out_tokens = out_tokens.to(images.dtype)
        return out_tokens, all_indices

# class CLIPVisionTower_SparseZip_EXP(nn.Module):

#     @torch.no_grad()
#     def forward(self, images):
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

#             feats = torch.stack(image_features, dim=0)  # [B, T, C]
#             self._last_keep_idx = None
#             return feats

#         outs = self.vision_tower(
#             images.to(device=self.device, dtype=self.dtype),
#             output_hidden_states=True,
#             output_attentions=True,
#         )
#         attn_weights  = outs.attentions[-2]       # [B, H, L, L]
#         hidden_states = outs.hidden_states[-2]    # [B, L, C]
#         metric = self.vision_tower.vision_model.encoder.layers[-2].metric  # [B, L, Ck]

#         scoring_layers = [{
#             "attn": attn_weights,   # [B, H, L, L]
#             "keys": metric,         # [B, L, Ck]
#         }]
#         cross_attn = None

#         if not hasattr(self, "_vz_comp") or self._vz_comp is None:
#             raise RuntimeError(
#                 "[SparseZip] _vz_comp not found on CLIPVisionTower. "
#                 "Make sure you build the model via LlavaSparseZipModel "
#                 "so that vt._vz_comp is attached."
#             )

#         dominant_num = None

#         out_tokens, all_indices = self._vz_comp(
#             scoring_layers=scoring_layers,
#             hidden_states=hidden_states,          # [B, L, C]
#             cross_attn=cross_attn,
#             cross_last_dim_is_L=True,
#             dominant_num=dominant_num,
#             weights_for_context=None,
#         )
#         # print("out_tokens shape:", out_tokens.shape)
#         # out_tokens: [B, K+1+ctx, C]
#         # all_indices: [B, <=L]

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

#         self._last_keep_idx = all_indices

        return out_tokens, all_indices
#         return out_tokens
