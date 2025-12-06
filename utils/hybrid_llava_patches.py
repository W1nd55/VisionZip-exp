# utils/hybrid_llava_patches.py
from __future__ import annotations

from typing import List, Any

import torch

from llava.model.llava_arch import LlavaMetaForCausalLM, unpad_image
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import get_anyres_image_grid_shape

def patch_llava_multimodal_for_hybrid():
    """
    Monkey-patch LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal
    so that when constructing new_input_embeds, it simultaneously constructs a
      self._hybrid_token_type_mask: BoolTensor [B, L]
    where 1 represents a vision token and 0 represents a text token.
    """

    if getattr(LlavaMetaForCausalLM, "_hybrid_patched", False):
        # Avoid duplicate patching
        print("[Hybrid] LlavaMetaForCausalLM already patched, skip.")
        return

    print("[Hybrid] Patching LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal for hybrid token mask.")

    def prepare_inputs_labels_for_multimodal_hybrid(
        self: LlavaMetaForCausalLM,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        """
        Copied from the original prepare_inputs_labels_for_multimodal.
        The only addition: while constructing new_input_embeds, maintain a 0/1 token_type_mask 
        and attach it to self._hybrid_token_type_mask before returning.
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # Pure text / decode step: do not construct mask
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        # -------- 1) Calculate image_features using original logic first --------
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [
                    x.unsqueeze(0) if x.ndim == 3 else x
                    for x in images
                ]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features: List[torch.Tensor] = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == "anyres":
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx],
                                self.config.image_grid_pinpoints,
                                self.get_vision_tower().config.image_size,
                            )
                            image_feature = image_feature.view(
                                num_patch_height,
                                num_patch_width,
                                height,
                                width,
                                -1,
                            )
                        else:
                            raise NotImplementedError
                        if "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat(
                            (base_image_feature, image_feature), dim=0
                        )
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[None].to(
                                        image_feature.device
                                    ),
                                ),
                                dim=0,
                            )
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}"
                )
        else:
            image_features = self.encode_images(images)

        # -------- 2) Preparation of text / mask / labels (Original) --------
        if (
            getattr(self.config, "tune_mm_mlp_adapter", False)
            and getattr(self.config, "mm_use_im_start_end", False)
        ):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0,
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(
                input_ids, attention_mask
            )
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        # -------- 3) For each sample, replace IMAGE_TOKEN_INDEX with image_features --------
        new_input_embeds: List[torch.Tensor] = []
        new_labels: List[torch.Tensor] = []
        new_token_type_masks: List[torch.Tensor] = []  # 0=text, 1=vision
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # No IMAGE_TOKEN_INDEX, follow original behavior: pure text + (optional) image_features[0:0]
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])

                # token_type_mask: All 0 (pure text)
                new_token_type_masks.append(
                    torch.zeros(
                        cur_input_embeds.shape[0],
                        dtype=torch.bool,
                        device=cur_input_embeds.device,
                    )
                )

                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim: List[torch.Tensor] = []
            cur_labels = labels[batch_idx]
            cur_labels_noim: List[torch.Tensor] = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(
                cur_input_embeds, split_sizes, dim=0
            )
            cur_new_input_embeds: List[torch.Tensor] = []
            cur_new_labels: List[torch.Tensor] = []
            cur_new_token_type_mask: List[torch.Tensor] = []

            for i in range(num_images + 1):
                # Text segment: mask=0
                text_segment = cur_input_embeds_no_im[i]
                text_labels = cur_labels_noim[i]
                cur_new_input_embeds.append(text_segment)
                cur_new_labels.append(text_labels)
                cur_new_token_type_mask.append(
                    torch.zeros(
                        text_segment.shape[0],
                        dtype=torch.bool,
                        device=text_segment.device,
                    )
                )

                if i < num_images:
                    # image_features segment: mask=1
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                    cur_new_token_type_mask.append(
                        torch.ones(
                            cur_image_features.shape[0],
                            dtype=torch.bool,
                            device=cur_image_features.device,
                        )
                    )

            # concat into a long sequence
            cur_new_input_embeds = [
                x.to(self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_token_type_mask = torch.cat(cur_new_token_type_mask)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_token_type_masks.append(cur_new_token_type_mask)

        # -------- 4) Truncate to tokenizer_model_max_length (Original) --------
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [
                x[:tokenizer_model_max_length] for x in new_labels
            ]
            new_token_type_masks = [
                x[:tokenizer_model_max_length]
                for x in new_token_type_masks
            ]

        # -------- 5) Pad into batch format --------
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded: List[torch.Tensor] = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask_out = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids_out = torch.zeros(
            (batch_size, max_len),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        token_type_mask_padded = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
            device=new_labels[0].device,
        )

        for i, (cur_new_embed, cur_new_labels, cur_new_tmask) in enumerate(
            zip(new_input_embeds, new_labels, new_token_type_masks)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask_out[i, -cur_len:] = True
                    position_ids_out[i, -cur_len:] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
                    token_type_mask_padded[i, -cur_len:] = cur_new_tmask
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask_out[i, :cur_len] = True
                    position_ids_out[i, :cur_len] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
                    token_type_mask_padded[i, :cur_len] = cur_new_tmask

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # Restoration of labels / attention_mask / position_ids (original logic)
        if _labels is None:
            new_labels_final = None
        else:
            new_labels_final = new_labels_padded

        if _attention_mask is None:
            attention_mask_final = None
        else:
            attention_mask_final = attention_mask_out.to(
                dtype=_attention_mask.dtype
            )

        if _position_ids is None:
            position_ids_final = None
        else:
            position_ids_final = position_ids_out

        # -------- 6) Attach token_type_mask to self for use by attention hook --------
        # Shape: [B, max_len], dtype: bool
        self._hybrid_token_type_mask = token_type_mask_padded

        return (
            None,
            position_ids_final,
            attention_mask_final,
            past_key_values,
            new_input_embeds,
            new_labels_final,
        )

    # Replace the original method
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = (
        prepare_inputs_labels_for_multimodal_hybrid
    )
    LlavaMetaForCausalLM._hybrid_patched = True