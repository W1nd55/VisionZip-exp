# utils/clip_encoder_sparsezip_exp.py
import torch
import torch.nn as nn
from typing import Optional, List, Dict

class CLIPVisionTower_SparseZip_EXP(nn.Module):

    @torch.no_grad()
    def forward(self, images):
        """
        self: 实际上是 llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower 的实例
              上面已经有:
                - self.vision_tower: HF 的 CLIPVisionModel
                - self.device / self.dtype
                - self._info["dominant"], self._info["contextual"]
                - （由 LlavaSparseZipModel 构造时挂上的）self._vz_comp: VisionZipCompressor
        """

        # 1) list 输入：一般是 chat/serve 那种多图模式，保持原始 feature_select，暂时不做 SparseZip
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

        # 2) batch 输入：真正走 SparseZip 压缩
        # images: [B,3,H,W]
        outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=True,
        )
        attn_weights  = outs.attentions[-2]       # [B,H,L,L]
        hidden_states = outs.hidden_states[-2]    # [B,L,C]
        # metric 是在 CLIPEncoderLayer.forward 里 monkey-patch 存进去的 key head-mean
        metric = self.vision_tower.vision_model.encoder.layers[-2].metric  # [B,L,Ck]

        # --- 准备给 compressor 的输入 ---
        scoring_layers = [{
            "attn": attn_weights,   # [B,H,L,L]
            "keys": metric,         # [B,L,Ck]
        }]

        # 可选 cross-attn，这里先没有
        cross_attn = None

        # 3) 取出挂在 vision tower 上的 compressor
        if not hasattr(self, "_vz_comp") or self._vz_comp is None:
            raise RuntimeError(
                "[SparseZip] _vz_comp not found on CLIPVisionTower. "
                "Make sure you build the model via LlavaSparseZipModel "
                "so that vt._vz_comp is attached."
            )

        # dominant K 的选择策略：
        #   - 如果想要“完全交给 dynamic-K”：dominant_num = None
        #   - 如果想固定成 cfg 里的 dominant：dominant_num = int(self._info['dominant'])
        # 你现在写的 VisionZipCompressor 已经有 dynamic_k 逻辑，所以这里推荐用 None，让它自己决定。
        dominant_num = None    # 或者 int(self._info["dominant"])

        # 4) 调用 SparseZip compressor：scores -> K dominant + hierarchical contextual merge
        out_tokens, all_indices = self._vz_comp(
            scoring_layers=scoring_layers,
            hidden_states=hidden_states,          # [B,L,C]
            cross_attn=cross_attn,
            cross_last_dim_is_L=True,
            dominant_num=dominant_num,
            weights_for_context=None,            # 后续如果你有额外权重，可以在这里加
        )
        # out_tokens: [B, K+1+ctx, C]
        # all_indices: [B, <=L]，原始 token 的 index（CLS+dominant，其余在 merge 时只参与聚合）

        out_tokens = out_tokens.to(images.dtype)
        return out_tokens, all_indices