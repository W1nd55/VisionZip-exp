from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional, Tuple

@dataclass
class HybridConfig:
    def __init__(
        self,
        prune_layers=None,
        dominant_tokens: int = 54,
        min_tokens: int = 16,
        alpha_min: float = 0.6,
        alpha_max: float = 0.9,
        drop_alpha: float = 0.3,
        beta: float = 0.0,
        n_min: int = 64,
        max_rank_k: int = 64,
    ):
        self.prune_layers = prune_layers if prune_layers else []
        self.dominant_tokens = int(dominant_tokens)
        self.min_tokens = int(min_tokens)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.drop_alpha = float(drop_alpha)
        self.beta = beta
        self.n_min = n_min
        self.max_rank_k = max_rank_k


def install_hybrid_on_llama(top_model, sparsifier):
    model = top_model.get_model() if hasattr(top_model, "get_model") else top_model

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        decoder_layers = model.model.layers
    elif hasattr(model, "layers"):
        decoder_layers = model.layers
    else:
        print("[Hybrid] Cannot find layers to patch.")
        return

    num_layers = len(decoder_layers)
    print(f"[Hybrid] Installing Robust Pruning (Residual+RoPE Safe) on {num_layers} layers.")

    def make_layer_forward(original_layer_instance, top_model, sparsifier, layer_idx):
        def hybrid_layer_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
        ):
            # ------------------------------------------------------------------
            # Part 1: Strict Input Sanitization
            # ------------------------------------------------------------------
            # Ensure hidden_states is clean
            if not hidden_states.is_contiguous():
                hidden_states = hidden_states.contiguous()
            
            B, T, C = hidden_states.shape
            device = hidden_states.device
            
            # --- CLEAN KWARGS ---
            # Remove any potentially conflicting arguments that LlamaAttention might grab
            for key in ["position_ids", "attention_mask", "cache_position"]:
                if key in kwargs:
                    del kwargs[key]

            # --- REGENERATE POSITION IDS ---
            # Always regenerate to ensure 0..T-1 mapping.
            # This is crucial for RoPE to not read out of bounds.
            new_pos = torch.arange(T, dtype=torch.long, device=device)
            new_pos = new_pos.unsqueeze(0).expand(B, -1)
            position_ids = new_pos.contiguous()

            # --- REGENERATE ATTENTION MASK ---
            # If T > 1, we MUST provide a valid causal mask, otherwise LLaMA might try to build one
            # using wrong assumptions about cache_position.
            if T > 1:
                # Standard Causal Mask: min_val above diagonal
                mask = torch.full((1, 1, T, T), torch.finfo(hidden_states.dtype).min, device=device, dtype=hidden_states.dtype)
                mask = torch.triu(mask, diagonal=1)
                attention_mask = mask.contiguous()
            else:
                # For decoding (T=1), usually mask is None or handled internally
                attention_mask = None

            # ------------------------------------------------------------------
            # Part 2: Pruning Logic Prep
            # ------------------------------------------------------------------
            token_type_mask = getattr(top_model, "_hybrid_token_type_mask", None)
            
            current_mask = None
            if token_type_mask is not None:
                # Slice mask to match current T
                current_mask = token_type_mask[:, :T].contiguous()
            
            should_prune = (
                current_mask is not None 
                and T > 1
                and layer_idx in sparsifier.cfg.prune_layers
            )

            # ------------------------------------------------------------------
            # Part 3: Manual Layer Execution
            # ------------------------------------------------------------------
            
            # 1. Norm
            residual = hidden_states
            hidden_states = original_layer_instance.input_layernorm(hidden_states)

            # 2. Self Attention
            # Force output_attentions=True if we are pruning, so we get the weights
            need_weights = should_prune or output_attentions
            
            # We explicitly pass our sanitized position_ids and attention_mask
            attn_outputs = original_layer_instance.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=need_weights,
                use_cache=use_cache,
                **kwargs
            )
            
            attn_output = attn_outputs[0]
            attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
            present_key_value = attn_outputs[2] if len(attn_outputs) > 2 else None

            # 3. Pruning Intervention
            if should_prune:
                # Run Sparsifier
                new_h, new_mask, keep_indices, info = sparsifier.forward(
                    attn_output,    # Use attn_output (post-attention features)
                    attn_weights,
                    current_mask,
                    layer_idx
                )
                
                if keep_indices is not None and keep_indices.shape[0] < T:
                    # A. Prune Attention Output
                    attn_output = new_h.contiguous()
                    
                    # B. Prune Residual (The specific fix for shape mismatch)
                    residual = residual[:, keep_indices, :].contiguous()
                    
                    # C. Update Global Mask (for next layers)
                    top_model._hybrid_token_type_mask = new_mask
                    
                    # D. Log
                    if layer_idx == sparsifier.cfg.prune_layers[0]:
                        old_vis = int(current_mask.sum().item())
                        new_vis = int(new_mask.sum().item())
                        # print(f"[HybridTrace] Layer {layer_idx} PRUNED: {T}->{new_h.shape[1]} (Vis {old_vis}->{new_vis})")

                    # E. Prune KV Cache (Crucial for subsequent decoding steps)
                    if present_key_value is not None:
                        k_tp, v_tp = None, None
                        is_dyn = hasattr(present_key_value, 'key_cache')
                        
                        if is_dyn:
                            # DynamicCache
                            if len(present_key_value.key_cache) > layer_idx:
                                k_tp = present_key_value.key_cache[layer_idx]
                                v_tp = present_key_value.value_cache[layer_idx]
                        elif isinstance(present_key_value, (tuple, list)):
                            # Standard Tuple
                            k_tp, v_tp = present_key_value

                        if k_tp is not None:
                            # Ensure safety
                            if not k_tp.is_contiguous(): k_tp = k_tp.contiguous()
                            if not v_tp.is_contiguous(): v_tp = v_tp.contiguous()
                            
                            kv_len = k_tp.shape[2]
                            max_idx = keep_indices.max().item() if keep_indices.numel() > 0 else 0
                            
                            if max_idx < kv_len:
                                kp = k_tp[:, :, keep_indices, :].contiguous()
                                vp = v_tp[:, :, keep_indices, :].contiguous()
                                
                                if is_dyn:
                                    present_key_value.key_cache[layer_idx] = kp
                                    present_key_value.value_cache[layer_idx] = vp
                                else:
                                    present_key_value = (kp, vp)

            # 4. Residual Add 1
            # Now both residual and attn_output have been pruned to the same length
            hidden_states = residual + attn_output

            # 5. MLP Block
            residual = hidden_states
            hidden_states = original_layer_instance.post_attention_layernorm(hidden_states)
            hidden_states = original_layer_instance.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)
            if output_attentions: outputs += (attn_weights,)
            if use_cache: outputs += (present_key_value,)

            return outputs

        return hybrid_layer_forward

    for idx, layer in enumerate(decoder_layers):
        layer.forward = make_layer_forward(layer, top_model, sparsifier, idx)

    print("[Hybrid] Layers fully hijacked successfully.")