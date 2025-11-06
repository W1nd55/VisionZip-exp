import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from typing import Any, Optional, Tuple, Union, List
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPVisionTransformer, CLIPEncoder
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput


def CLIPAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    bsz, tgt_len, embed_dim = hidden_states.size()

    # Helper function to replace _shape method (for compatibility with newer transformers)
    def _shape(tensor, seq_len, bsz):
        """Reshape tensor for multi-head attention"""
        if seq_len == -1:
            # Infer sequence length from tensor shape
            seq_len = tensor.size(1)
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scale
    key_states = _shape(self.k_proj(hidden_states), -1, bsz)
    raw_key_states = key_states.clone()
    value_states = _shape(self.v_proj(hidden_states), -1, bsz)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = _shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    # apply the causal_attention_mask first
    if causal_attention_mask is not None:
        if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                f" {causal_attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)



    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if output_attentions:
        # this operation is a bit akward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped, raw_key_states.mean(1)

def CLIP_EncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            `(config.encoder_attention_heads,)`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
    """
    residual = hidden_states

    hidden_states = self.layer_norm1(hidden_states)


    # Call the patched CLIPAttention forward - this should return 3 values: (attn_output, attn_weights, metric)
    attn_output = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )
    
    # Unpack the result - handle both 2-tuple (original) and 3-tuple (patched) returns
    if isinstance(attn_output, tuple):
        if len(attn_output) == 3:
            hidden_states, attn_weights, metric = attn_output
        elif len(attn_output) == 2:
            hidden_states, attn_weights = attn_output
            metric = None  # Original CLIPAttention doesn't return metric
            # DEBUG: This means the patching didn't work!
            layer_idx = self._info.get("layer_idx", -1) if hasattr(self, '_info') else -1
            if layer_idx == 22:
                import warnings
                warnings.warn(f"CRITICAL: Layer {layer_idx} received 2-tuple from CLIPAttention.forward - patching may have failed!")
        else:
            raise ValueError(f"Unexpected number of return values from self_attn: {len(attn_output)}")
    else:
        # Single return value (shouldn't happen with CLIPAttention)
        hidden_states = attn_output
        attn_weights = None
        metric = None
    
    hidden_states = residual + hidden_states
    
    # Get r value from _info based on layer index - CRITICAL: This must work correctly
    layer_idx = self._info.get("layer_idx", -1) if hasattr(self, '_info') else -1
    r_list = self._info.get("r", []) if hasattr(self, '_info') else []
    
    # CRITICAL FIX: Ensure metric is set correctly for layer 22 (index 22, r=1)
    # The metric comes from CLIPAttention_forward's raw_key_states.mean(1)
    if isinstance(r_list, list) and len(r_list) > layer_idx and layer_idx >= 0:
        # Use layer index to get the correct r value
        r = r_list[layer_idx]
        # Only set metric if r > 0 (this should be layer 22, index 22)
        if r > 0:
            # Store metric on the layer itself - CRITICAL: Make sure it's a tensor, not None
            if metric is not None:
                self.metric = metric.detach() if hasattr(metric, 'detach') else metric
                # Also store on the encoder module for reliable access (multi-GPU safe)
                # We stored the encoder reference in _info during apply_info
                try:
                    encoder = self._info.get("_encoder", None)
                    if encoder is not None:
                        encoder._visionzip_metric = metric.detach() if hasattr(metric, 'detach') else metric
                except Exception as e:
                    # If we can't store on parent, that's okay - metric is on self
                    pass
            else:
                # CRITICAL ERROR: metric is None - this means CLIPAttention_forward patching failed!
                import warnings
                warnings.warn(
                    f"CRITICAL: Layer {layer_idx} has r={r} but metric is None! "
                    f"This means CLIPAttention.forward is not returning 3 values. "
                    f"Patching may have failed. Check that CLIPAttention.forward was patched correctly."
                )
    else:
        # Fallback detection: if layer_idx is missing, check if we're the second-to-last layer
        # With 24 layers, index 22 (second-to-last) should have r=1
        if layer_idx == 22:
            self.metric = metric
        elif layer_idx == -1:
            # Try to infer: if r_list has 24 elements and index 22 is 1, we might be layer 22
            if isinstance(r_list, list) and len(r_list) == 24 and r_list[22] == 1:
                # Check if we're actually the second-to-last layer by counting
                # This is a fallback - _info should always have layer_idx
                try:
                    # Try to find our position in the encoder
                    encoder = getattr(self, '__self__', None)
                    if encoder is None:
                        # Try parent
                        parent = getattr(self, 'parent', None)
                        if parent and hasattr(parent, 'layers'):
                            encoder = parent
                    if encoder and hasattr(encoder, 'layers'):
                        layers = encoder.layers
                        if isinstance(layers, (list, tuple)) and len(layers) >= 2:
                            # Check if we're the second-to-last
                            if self is layers[-2] or (hasattr(layers[-2], 'id') and hasattr(self, 'id') and layers[-2].id == self.id):
                                self.metric = metric
                except:
                    pass
    
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Copy from the TOME. 
    https://github.com/facebookresearch/ToMe

    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]

def make_tome_class(transformer_class):
    class VisionZipTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
            
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._info["r"] = parse_r(len(self.vision_model.encoder.layers), self.r)
            # self._info["r"] = self.r

            self._info["size"] = None
            self._info["source"] = None

            return super().forward(*args, **kwdargs)

    return VisionZipTransformer

def apply_info(model, dominant_num, contextual_num):

    VisionZipTransformer = make_tome_class(model.__class__)

    model.__class__ = VisionZipTransformer
    # Create r list: 22 zeros, then 1, then 0 (total 24 elements for 24 layers)
    model.r = [0 for i in range(22)]+ [1]+[0]

    model._info = {
        "r": model.r,  # Store the r list
        "dominant":dominant_num,
        "contextual":contextual_num,
    }
    
    # Find the encoder module and all encoder layers
    encoder = None
    encoder_layers = []
    for name, module in model.named_modules():
        if isinstance(module, CLIPEncoderLayer):
            encoder_layers.append(module)
        elif isinstance(module, CLIPEncoder):
            encoder = module
    
    # Verify we have the expected number of layers (should be 24 for CLIP)
    expected_layers = 24
    if len(encoder_layers) != expected_layers:
        print(f"   ⚠ Warning: Found {len(encoder_layers)} encoder layers, expected {expected_layers}")
    
    if encoder is None:
        print(f"   ⚠ Warning: Could not find CLIPEncoder module")
    
    # Now assign _info to each layer with its index - CRITICAL: This must be correct
    for idx, module in enumerate(encoder_layers):
        # Each layer gets its own copy of r, and we track which layer this is
        module._info = {
            "r": model.r.copy(),  # Copy the list for each layer
            "layer_idx": idx,  # Track which layer this is (0-23 for 24 layers)
            "dominant": dominant_num,
            "contextual": contextual_num,
            "_encoder": encoder,  # Store reference to encoder for metric storage
        }
        # Verify layer 22 has r=1
        if idx == 22:
            r_val = model.r[idx] if idx < len(model.r) else 0
            if r_val != 1:
                print(f"   ⚠ CRITICAL: Layer {idx} has r={r_val}, expected r=1")
            else:
                print(f"   ✓ Layer {idx} correctly configured with r=1")

