import math
from typing import Optional, Tuple
import torch

# =============================== #
# Text-Aware VisionZip Components #
# =============================== #

class TextContext:
    """
    Global context used to pass text features between LLaVA's prepare_inputs and CLIP's forward.
    This is a simple singleton pattern implementation.
    """
    _current_text_query = None
    _text_projector = None # This is an nn.Linear, needs to be registered into the model

    @classmethod
    def set_text_query(cls, query_tensor):
        cls._current_text_query = query_tensor

    @classmethod
    def get_text_query(cls):
        return cls._current_text_query

    @classmethod
    def set_projector(cls, projector_module):
        cls._text_projector = projector_module

    @classmethod
    def get_projector(cls):
        return cls._text_projector

def bipartite_soft_matching_random2d(metric, w, r, protected, random=False):
    """
    Core merging logic for VisionZip/ToMe (Simplified reference for custom Forward).
    If your visionzip package has similar utils, you can import them directly.
    A standard implementation is provided here for independence.
    """
    B, N, _ = metric.shape
    if r <= 0: return protected, {}

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if random:
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        else:
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
            unm_idx = edge_idx[..., r:, :]  # Unmerged
            src_idx = edge_idx[..., :r, :]  # Merged
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    # Returns merged index information. The specific implementation is slightly complex.
    # Here, to demonstrate the core Text-Aware logic, we assume using the merge logic 
    # included in the visionzip library.
    # The key lies in the construction of 'metric' (i.e., features) above.
    return src_idx, dst_idx # Demonstration only


def text_aware_visionzip_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor]:
    
    batch_size, seq_len, embed_dim = hidden_states.shape
    
    # 1. Prepare Residual
    residual = hidden_states

    # 2. Layer Norm 1
    hidden_states = self.layer_norm1(hidden_states)
    
    layer_idx = getattr(self, "_layer_idx", -1)
    # print(f"[TA] Layer {layer_idx} before prune: {hidden_states.shape[1]}") # Optional: reduce log spam

    # ==========================================================
    # Text-Aware Pruning Logic (Target-Oriented)
    # ==========================================================
    vis_alpha = getattr(self, "vis_alpha", 0.6)
    
    # The floor here is what we set in __init__ (dominant + 16), e.g., 54+16=70
    target_token_num = getattr(self, "min_tokens_floor", 70) 

    # [Gate] As long as current count is greater than target, we prune!
    if seq_len > target_token_num:
        
        # --- A. Separate [CLS] ---
        cls_token = hidden_states[:, :1, :]
        patch_tokens = hidden_states[:, 1:, :]
        
        cls_residual = residual[:, :1, :]
        patch_residual = residual[:, 1:, :]
        
        # --- [Key Modification] Calculate num_keep ---
        # Goal is to let (1 CLS + Kept Patches) = target_token_num
        # So the number of patches to keep = target_token_num - 1
        num_keep = target_token_num - 1
        
        # Check again to prevent target from being smaller than 1 (though unlikely)
        if num_keep < 1: num_keep = 1
        
        current_patch_num = patch_tokens.shape[1]

        # Only perform calculation and pruning if existing patches are strictly more than we want to keep
        if current_patch_num > num_keep:
            
            # --- B. Calculate Scores (Vision + Text) ---
            states_for_score = patch_tokens.detach()
            norm_states = states_for_score / (states_for_score.norm(dim=-1, keepdim=True) + 1e-6)
            
            # Vision Score
            sim_matrix = torch.bmm(norm_states, norm_states.transpose(1, 2))
            vis_score = sim_matrix.mean(dim=1)
            
            # Text Score
            text_ctx = TextContext.get_text_query()
            projector = TextContext.get_projector()
            
            final_score = vis_score
            if text_ctx is not None and projector is not None:
                text_ctx = text_ctx.to(dtype=hidden_states.dtype, device=hidden_states.device)
                text_query_vis = projector(text_ctx)
                text_relevance = torch.bmm(states_for_score, text_query_vis.transpose(1, 2)).squeeze(-1)
                text_relevance = torch.sigmoid(text_relevance)
                
                v_min, v_max = vis_score.min(dim=1, keepdim=True)[0], vis_score.max(dim=1, keepdim=True)[0]
                vis_score_norm = (vis_score - v_min) / (v_max - v_min + 1e-6)
                final_score = vis_alpha * vis_score_norm + (1 - vis_alpha) * text_relevance

            # --- C. Execute Pruning (One-step) ---
            # Directly select Top-K (K=num_keep)
            topk_scores, topk_indices = torch.topk(final_score, num_keep, dim=1)
            
            # Gather
            gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
            patch_tokens = torch.gather(patch_tokens, 1, gather_idx)
            patch_residual = torch.gather(patch_residual, 1, gather_idx)
            
            # --- D. Reassemble ---
            hidden_states = torch.cat([cls_token, patch_tokens], dim=1)
            residual = torch.cat([cls_residual, patch_residual], dim=1)
            
            # Reset Masks
            attention_mask = None
            causal_attention_mask = None

    # ==========================================================
    # Update self.metric
    # ==========================================================
    self.metric = hidden_states 

    # ==========================================================
    # Self-Attention
    # ==========================================================
    attn_outputs = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )
    
    attn_weights = None
    if isinstance(attn_outputs, tuple):
        hidden_states = attn_outputs[0]
        if len(attn_outputs) > 1:
            attn_weights = attn_outputs[1]
    else:
        hidden_states = attn_outputs

    # 4. Residual Connection 1
    hidden_states = residual + hidden_states

    # 5. MLP Block
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    
    # 6. Residual Connection 2
    hidden_states = residual + hidden_states
    
    # print(f"[TA] Layer {layer_idx} after prune: {hidden_states.shape[1]}") # Expected output should be around 70

    # Return
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)
        
    return outputs