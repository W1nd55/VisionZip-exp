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

#         self._last_keep_idx = all_indices

#         return out_tokens