from typing import Any, Dict, Tuple, List, Optional, Union
import time
from PIL import Image
import torch
from scripts.timer import StageTimer
from scripts.abstract import BaseModel, Sample
import gc
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
visionzip_parent_dir = os.path.join(current_dir, os.pardir, 'models/VisionZip')
if visionzip_parent_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(visionzip_parent_dir))

# ========================= #
# Model: LLaVA + VisionZip  #
# ========================= #
class LlavaModel(BaseModel):
    def __init__(self, model_path: str, temperature: float=0.2, max_new_tokens: int=256):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        )
        from llava.conversation import conv_templates

        # extra_kwargs = {
        #     "device_map": "auto",
        #     "torch_dtype": torch.float16,
        # }
        # for k, v in [
        #     ({"load_4bit": True,
        #       "bnb_4bit_compute_dtype": torch.float16,
        #       "bnb_4bit_quant_type": "nf4"}, "4bit"),
        #     ({"load_8bit": True}, "8bit"),
        #     ({}, "fp16")
        # ]:
        #     try:
        #         kwargs = extra_kwargs | k
        #         tokenizer, model, image_processor, _ = load_pretrained_model(
        #             model_path=model_path,
        #             model_base=None,
        #             model_name=get_model_name_from_path(model_path),
        #             **kwargs
        #         )
        #         print(f"[load] success with {v}")
        #         break
        #     except Exception as e:
        #         print(f"[load] try {v} failed:", e)
        #         continue
        
        model_name = get_model_name_from_path(model_path)

        print("[load] using pure fp16 on single GPU ...")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device_map=None, 
            torch_dtype=torch.float16,
        )
        print("[load] fp16 load success")

        # ---- move whole model to GPU/CPU ----
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model.to(device)
        self._device = device
        print(f"[load] model moved to {self._device}")

        # (Optional) override CLIPVisionTower.forward if you have a custom EXP version
        # try:
        #     from utils.clip_encoder_exp import CLIPVisionTower_VisionZip_EXP
        #     from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
        #     # Patching the CLIP Vision Tower forward method
        #     CLIPVisionTower.forward = CLIPVisionTower_VisionZip_EXP.forward
        # except Exception as e:
        #     print("[Warn] EXP forward not found or failed to patch:", e)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_templates = conv_templates
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        # Initialize the tokenizer_image_token function with a fallback mechanism
        self.tokenizer_image_token = self._make_tokenizer_image_token()

        self.model.eval()

    def _make_tokenizer_image_token(self):
        """Construct a tokenizer_image_token function with fallback."""
        try:
            # Try to import the original function from the installed llava package
            from llava.mm_utils import tokenizer_image_token as _tik

            def _wrap(prompt: str) -> torch.Tensor:
                return _tik(
                    prompt,
                    self.tokenizer,
                    self.IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
            return _wrap
        except Exception:
            # Fallback implementation if llava.mm_utils is not importable
            print("[Warn] Could not import tokenizer_image_token, using local fallback.")
            tok = self.tokenizer
            try:
                tok_img = tok.convert_tokens_to_ids(self.DEFAULT_IMAGE_TOKEN)
            except Exception:
                tok_img = None
            try:
                tok_im_start = tok.convert_tokens_to_ids(self.DEFAULT_IM_START_TOKEN)
            except Exception:
                tok_im_start = None
            try:
                tok_im_end = tok.convert_tokens_to_ids(self.DEFAULT_IM_END_TOKEN)
            except Exception:
                tok_im_end = None

            def _local(prompt: str) -> torch.Tensor:
                # Tokenize the prompt first
                ids = tok(prompt, return_tensors=None, add_special_tokens=True)["input_ids"]
                new_ids = []
                # Replace image placeholder tokens with the special IMAGE_TOKEN_INDEX
                for t in ids:
                    if tok_img is not None and t == tok_img:
                        new_ids.append(self.IMAGE_TOKEN_INDEX)
                    elif tok_im_start is not None and t == tok_im_start:
                        continue # Skip start token (if present and configured)
                    elif tok_im_end is not None and t == tok_im_end:
                        continue # Skip end token (if present and configured)
                    else:
                        new_ids.append(t)
                return torch.tensor(new_ids, dtype=torch.long)
            return _local

    def device(self) -> torch.device:
        """Returns the device of the model."""
        return self.model.device

    @torch.inference_mode()
    def prepare_inputs(self, sample: Sample) -> Dict[str, Any]:
        """Prepare inputs for a single sample.
        
        Returns dict with keys:
          - 'input_ids': torch.LongTensor
          - 'images': torch.FloatTensor or None
          - 'load_ms': float
          - 'preprocess_ms': float
        """
        # Image loading and preprocessing
        image_tensor = None
        t0 = time.perf_counter()
        img = Image.open(sample.image_path).convert("RGB") if sample.image_path else None
        load_ms = (time.perf_counter() - t0)*1000.0

        t1 = time.perf_counter()
        if img is not None:
            # Preprocess image into a tensor
            image_tensor = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'].half()
            image_tensor = image_tensor.to(self.device())
        preprocess_ms = (time.perf_counter() - t1)*1000.0

        # Prompt construction
        conv = self.conv_templates["llava_v1"].copy()
        image_placeholder = self.DEFAULT_IMAGE_TOKEN
        # Handle cases where the model expects start/end tokens around the image token
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_placeholder = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
        
        # Format the full input string
        full_user_input = image_placeholder + "\n" + sample.prompt
        conv.append_message(conv.roles[0], full_user_input)
        conv.append_message(conv.roles[1], None) # Append an empty assistant message to complete the prompt
        final_prompt_string = conv.get_prompt()

        # Tokenize the final prompt string using the custom/wrapped function
        ids = self.tokenizer_image_token(final_prompt_string)
        # Ensure it's a tensor before unsqueezing
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
            
        input_ids = ids.unsqueeze(0).to(self.device())

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "load_ms": load_ms,
            "preprocess_ms": preprocess_ms,
        }

    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generates the response and collects timing metrics."""
        timer = StageTimer(use_cuda=(self.device().type == "cuda"))

        # === Coarse-grained timing ===
        timer.start("end2end")
        timer.start("decode")  # Treating the entire generation as the decode phase for coarse timing

        out = self.model.generate(
            inputs=inputs["input_ids"],
            images=inputs["images"],
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
        )

        timer.end("decode")
        timer.end("end2end")

        out_ids = out.sequences
        # Decode the generated token IDs
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        # Calculate number of new tokens generated (compatible with both return styles)
        try:
            in_len  = inputs["input_ids"].shape[1]
            seq_len = out_ids.shape[1]
            gen_len = int(seq_len - in_len) if seq_len >= in_len else int(seq_len)
        except Exception:
            # Fallback for token count calculation
            gen_len = len(self.tokenizer(text, add_special_tokens=False).input_ids)

        timings_ms = {
            "load_ms":       inputs.get("load_ms", 0.0),
            "preprocess_ms": inputs.get("preprocess_ms", 0.0),
            "prefill_ms":    0.0,  # Coarse timing, prefill not separated
            "decode_ms":     timer.result_ms("decode"),
            "end2end_ms":    timer.result_ms("end2end"),
            "num_new_tokens": int(max(gen_len, 0)),
        }
        return {"text": text, **timings_ms}
    
class LlavaSparseZipModel(BaseModel):
    """
    Wrap your existing single-sample pipeline; keep it deterministic & timed.
    """
    def __init__(self, model_path: str, dominant: int=54, contextual: int=10, temperature: float=0.2, max_new_tokens: int=256, sparsezip_cfg: Optional[dict] = None):
        # Imports necessary for model loading and processing
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        # Removed: from llava.mm_utils import tokenizer_image_token (moved to _make_tokenizer_image_token)
        from llava.conversation import conv_templates
        
        # ---- Monkey patches (must be before loading model) ----
        # Patching CLIP attention/encoder layers. If external 'visionzip' utils are absent, skip gracefully.
        from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer
        try:
            from visionzip.utils import CLIPAttention_forward, CLIP_EncoderLayer_forward  # type: ignore
            CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
            CLIPAttention.forward = CLIPAttention_forward
        except Exception:
            print("[info] visionzip.utils not available; proceeding without low-level CLIP patch.")

        # (Optional) override CLIPVisionTower.forward if you have a custom EXP version
        try:
            # Prefer the new SparseZip compressor with dynamic-K, optional cross-attn, and hierarchical merge
            from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
            from utils.sparsezip import (
                VisionZipCompressor,
                CompressionConfig,
                MergingConfig,
                ScoringAlphas,
            )

            @torch.no_grad()
            def _sparsezip_forward(self, images):
                """
                Replacement forward for CLIPVisionTower implementing SparseZip compression.

                Returns:
                  hidden_states_save: FloatTensor [B, 1 + K + Ctx, C] (zero-padded per batch if dynamic K)
                  all_indices: LongTensor [B, 1 + K] (CLS + dominant), padded with -1 if variable-K
                """
                # If list of images, keep baseline behavior: feature_select per image (no compression)
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

                # Batched path with compression
                outs = self.vision_tower(
                    images.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True,
                    output_attentions=True,
                )

                # Use the second-to-last layer as in prior EXP implementation
                attn_weights = outs.attentions[-2]       # [B, H, L, L]
                hidden_states = outs.hidden_states[-2]   # [B, L, C]
                # Keys captured by prior monkey-patching (mean over heads), provided by CLIP encoder layer
                keys = self.vision_tower.vision_model.encoder.layers[-2].metric  # [B, L, Ck]

                # Read contextual count from VisionZip info (fallback to YAML sparsezip.merging.contextual_num or ctor arg)
                contextual_num = None
                try:
                    contextual_num = int(self.vision_tower._info.get("contextual", 16))
                except Exception:
                    contextual_num = 16

                # Compressor now pre-built in model __init__; update contextual target dynamically if needed
                if hasattr(self, "_vz_comp"):
                    try:
                        self._vz_comp.cfg.merging.contextual_num = contextual_num
                    except Exception:
                        pass

                # Optional weights for attention-weighted merge: CLS->patch mean attention
                weights_for_context = attn_weights[:, :, 0, 1:].mean(dim=1)  # [B, L-1]

                # Run compression (dynamic K by default; dominant_num=None)
                tokens, all_indices = self._vz_comp(
                    scoring_layers=[{"attn": attn_weights, "keys": keys}],
                    hidden_states=hidden_states,
                    cross_attn=None,
                    cross_last_dim_is_L=True,
                    dominant_num=None,
                    weights_for_context=weights_for_context,
                )

                return tokens.to(images.dtype), all_indices

            # Patch in the new forward
            # Stash YAML sparsezip cfg for the patched forward (bound to CLIPVisionTower instance at runtime)
            # We attach to the parent LlavaVisionZipModel instance so the closure can access it via 'self._sparsezip_cfg'
            # Note: After model loading below, we'll set model.mm_projector.vision_tower._sparsezip_cfg if present.
            CLIPVisionTower.forward = _sparsezip_forward
        except Exception as e:
            print("[Warn] SparseZip forward patch failed, falling back to EXP forward:", e)
            try:
                from utils.clip_encoder_exp import CLIPVisionTower_VisionZip_EXP
                from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower as _CT
                _CT.forward = CLIPVisionTower_VisionZip_EXP.forward
            except Exception as ee:
                print("[Warn] EXP forward not found or failed to patch:", ee)

        # ---- Load model ----
        import platform
        # Predeclare to allow fallback logic if all attempts fail
        tokenizer = None
        model = None
        image_processor = None

        is_macos = platform.system() == "Darwin"
        # On macOS / Apple Silicon we avoid device_map="auto" because offloading + fp16 can fail; fall back to eager load on single device
        extra_kwargs = {
            "device_map": None if is_macos else "auto",
            "torch_dtype": torch.float16,
            "attn_implementation": "eager",
        }
        quant_attempts: List[Tuple[Dict[str, Any], str]]
        if is_macos:
            # bitsandbytes not supported; fp16 first, then float32 fallback
            quant_attempts = [({}, "fp16"), ( {"torch_dtype": torch.float32}, "fp32")]
        else:
            quant_attempts = [
                ({"load_4bit": True,
                  "bnb_4bit_compute_dtype": torch.float16,
                  "bnb_4bit_quant_type": "nf4"}, "4bit"),
                ({"load_8bit": True}, "8bit"),
                ({}, "fp16"),
            ]
        for attempt_overrides, tag in quant_attempts:
            try:
                # Allow attempt to override torch_dtype if provided (e.g., fp32 fallback)
                merged = extra_kwargs | attempt_overrides
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=get_model_name_from_path(model_path),
                    **merged
                )
                print(f"[load] success with {tag}")
                break
            except Exception as e:
                print(f"[load] try {tag} failed:", e)
                tokenizer = model = image_processor = None
                continue

        # Final fallback: attempt plain float32 single-device load if still not loaded
        if model is None:
            try:
                print("[load] attempting final float32 fallback (no quant, no device_map)")
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=get_model_name_from_path(model_path),
                    device_map=None,
                    torch_dtype=torch.float32,
                )
                print("[load] success with final fp32 fallback")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model '{model_path}' after all quantization / dtype attempts. Last error: {e}\n"
                    "Suggestions: (1) Ensure sufficient RAM/VRAM; (2) Try a smaller checkpoint; (3) Update torch + transformers; (4) Remove any incompatible quant flags."
                )
        
        # Remove dependency on external 'visionzip' wrapper (not present in repo).
        # Instead, we simply stash dominant/contextual on a lightweight info dict for later reference.
        if not hasattr(model, "_vz_info"):
            model._vz_info = {"dominant": dominant, "contextual": contextual}

        # Attach sparsezip config on the actual CLIP vision tower instance for the patched forward to read
        try:
            vt = getattr(model, "mm_projector", None)
            vt = getattr(vt, "vision_tower", None)
            if vt is not None:
                # Stash dominant/contextual on the vision tower as well for local access
                try:
                    setattr(vt, "_info", {"dominant": dominant, "contextual": contextual})
                except Exception:
                    pass

                # Propagate YAML config for SparseZip
                if isinstance(sparsezip_cfg, dict):
                    setattr(vt, "_sparsezip_cfg", sparsezip_cfg)

                # Build and attach a VisionZipCompressor instance here (decoupled from forward)
                try:
                    from utils.sparsezip import (
                        VisionZipCompressor,
                        CompressionConfig,
                        MergingConfig,
                        ScoringAlphas,
                    )
                    _cfg = getattr(vt, "_sparsezip_cfg", {}) or {}
                    _alphas = _cfg.get("alphas", {}) if isinstance(_cfg, dict) else {}
                    contextual_num_cfg = int(((_cfg.get("merging", {}) or {}).get("contextual_num", contextual)))
                    cfg = CompressionConfig(
                        alphas=ScoringAlphas(
                            attn=float(_alphas.get("attn", 1.0)),
                            entropy=float(_alphas.get("entropy", 0.4)),
                            mutual=float(_alphas.get("mutual", 0.6)),
                        ),
                        tau_feat=float((_cfg.get("tau_feat", 0.2) if isinstance(_cfg, dict) else 0.2)),
                        tau_sim=float((_cfg.get("tau_sim", 0.1) if isinstance(_cfg, dict) else 0.1)),
                        cross_beta=float((_cfg.get("cross_beta", 0.0) if isinstance(_cfg, dict) else 0.0)),
                        dynamic_k=bool((_cfg.get("dynamic_k", True) if isinstance(_cfg, dict) else True)),
                        dynk=CompressionConfig().dynk.__class__(
                            c=float(((_cfg.get("dynk", {}) or {}).get("c", 8.0))),
                            eps=float(((_cfg.get("dynk", {}) or {}).get("eps", 1e-6))),
                            k_min=int((_cfg.get("k_min", 4))),
                            k_max=int((_cfg.get("k_max", 64))),
                        ),
                        k_min=int((_cfg.get("k_min", 4) if isinstance(_cfg, dict) else 4)),
                        k_max=int((_cfg.get("k_max", 64) if isinstance(_cfg, dict) else 64)),
                        merging=MergingConfig(
                            contextual_num=int(contextual_num_cfg),
                            kmeans_init_factor=float(((_cfg.get("merging", {}) or {}).get("kmeans_init_factor", 2.0))),
                            kmeans_iters=int(((_cfg.get("merging", {}) or {}).get("kmeans_iters", 10))),
                            agglomerative=bool(((_cfg.get("merging", {}) or {}).get("agglomerative", True))),
                        ),
                    )
                    setattr(vt, "_vz_comp", VisionZipCompressor(num_scoring_layers=1, cfg=cfg))
                except Exception as e:
                    print("[Warn] Could not construct VisionZipCompressor in __init__:", e)
        except Exception:
            pass

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_templates = conv_templates
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer_image_token = self._make_tokenizer_image_token()

        self.model.eval()

    # ---------------- Additional helper for SparseZip smoke testing ---------------- #
    @torch.inference_mode()
    def encode_image(self, pil_image: Image.Image):
        """Encodes an image through patched CLIP vision tower and returns (compressed_embeds, stats_dict).

        stats_dict keys:
          - k_dominant
          - c_contextual
          - orig_tokens
          - compressed_tokens
          - retained_indices (list)
        """
        # Preprocess
        img_tensor = self.image_processor.preprocess(pil_image, return_tensors='pt')['pixel_values'].to(self.device(), dtype=torch.float16)
        vt = getattr(self.model, 'mm_projector', None)
        vt = getattr(vt, 'vision_tower', None)
        stats = {}
        if vt is None:
            raise RuntimeError("Vision tower not found on model; cannot encode image.")
        # Run patched forward (returns tokens, indices)
        toks, indices = vt.forward(img_tensor)
        if indices is not None:
            # indices: [B, 1+K] with CLS + dominant indices (excluding contextual)
            k_dom = indices.shape[1] - 1
            c_ctx = getattr(vt, '_vz_comp', None)
            try:
                c_ctx = vt._vz_comp.cfg.merging.contextual_num  # type: ignore
            except Exception:
                c_ctx = -1
            stats = {
                'k_dominant': k_dom,
                'c_contextual': c_ctx,
                'orig_tokens': toks.shape[1],  # includes CLS + K + C
                'compressed_tokens': toks.shape[1],
                'retained_indices': indices[0].tolist(),
            }
        return toks, stats

    def _make_tokenizer_image_token(self):
        """Construct a tokenizer_image_token function with fallback."""
        try:
            # Try to import the original function from the installed llava package
            from llava.mm_utils import tokenizer_image_token as _tik

            def _wrap(prompt: str) -> torch.Tensor:
                return _tik(
                    prompt,
                    self.tokenizer,
                    self.IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
            return _wrap
        except Exception:
            # Fallback implementation if llava.mm_utils is not importable
            print("[Warn] Could not import tokenizer_image_token, using local fallback.")
            tok = self.tokenizer
            try:
                tok_img = tok.convert_tokens_to_ids(self.DEFAULT_IMAGE_TOKEN)
            except Exception:
                tok_img = None
            try:
                tok_im_start = tok.convert_tokens_to_ids(self.DEFAULT_IM_START_TOKEN)
            except Exception:
                tok_im_start = None
            try:
                tok_im_end = tok.convert_tokens_to_ids(self.DEFAULT_IM_END_TOKEN)
            except Exception:
                tok_im_end = None

            def _local(prompt: str) -> torch.Tensor:
                # Tokenize the prompt first
                ids = tok(prompt, return_tensors=None, add_special_tokens=True)["input_ids"]
                new_ids = []
                # Replace image placeholder tokens with the special IMAGE_TOKEN_INDEX
                for t in ids:
                    if tok_img is not None and t == tok_img:
                        new_ids.append(self.IMAGE_TOKEN_INDEX)
                    elif tok_im_start is not None and t == tok_im_start:
                        continue # Skip start token (if present and configured)
                    elif tok_im_end is not None and t == tok_im_end:
                        continue # Skip end token (if present and configured)
                    else:
                        new_ids.append(t)
                return torch.tensor(new_ids, dtype=torch.long)
            return _local

    def device(self) -> torch.device:
        """Returns the device of the model."""
        return self.model.device

    @torch.inference_mode()
    def prepare_inputs(self, sample: Sample) -> Dict[str, Any]:
        """Prepare inputs for a single sample.
        
        Returns dict with keys:
          - 'input_ids': torch.LongTensor
          - 'images': torch.FloatTensor or None
          - 'load_ms': float
          - 'preprocess_ms': float
        """
        # Image loading and preprocessing
        image_tensor = None
        t0 = time.perf_counter()
        img = Image.open(sample.image_path).convert("RGB") if sample.image_path else None
        load_ms = (time.perf_counter() - t0)*1000.0

        t1 = time.perf_counter()
        if img is not None:
            # Preprocess image into a tensor
            image_tensor = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'].half()
            image_tensor = image_tensor.to(self.device())
        preprocess_ms = (time.perf_counter() - t1)*1000.0

        # Prompt construction
        conv = self.conv_templates["llava_v1"].copy()
        image_placeholder = self.DEFAULT_IMAGE_TOKEN
        # Handle cases where the model expects start/end tokens around the image token
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_placeholder = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
        
        # Format the full input string
        full_user_input = image_placeholder + "\n" + sample.prompt
        conv.append_message(conv.roles[0], full_user_input)
        conv.append_message(conv.roles[1], None) # Append an empty assistant message to complete the prompt
        final_prompt_string = conv.get_prompt()

        # Tokenize the final prompt string using the custom/wrapped function
        ids = self.tokenizer_image_token(final_prompt_string)
        # Ensure it's a tensor before unsqueezing
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
            
        input_ids = ids.unsqueeze(0).to(self.device())

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "load_ms": load_ms,
            "preprocess_ms": preprocess_ms,
        }

    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generates the response and collects timing metrics."""
        timer = StageTimer(use_cuda=(self.device().type == "cuda"))

        # === Coarse-grained timing ===
        timer.start("end2end")

        out = self.model.generate(
            inputs=inputs["input_ids"],
            images=inputs["images"],
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
        )

        timer.end("end2end")

        out_ids = out.sequences
        # Decode the generated token IDs
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        # Calculate number of new tokens generated (compatible with both return styles)
        try:
            in_len  = inputs["input_ids"].shape[1]
            seq_len = out_ids.shape[1]
            gen_len = int(seq_len - in_len) if seq_len >= in_len else int(seq_len)
        except Exception:
            # Fallback for token count calculation
            gen_len = len(self.tokenizer(text, add_special_tokens=False).input_ids)

        timings_ms = {
            "load_ms":       inputs.get("load_ms", 0.0),
            "preprocess_ms": inputs.get("preprocess_ms", 0.0),
            "end2end_ms":    timer.result_ms("end2end"),
            "num_new_tokens": int(max(gen_len, 0)),
        }
        return {"text": text, **timings_ms}
    
class LlavaVisionZipModel(LlavaModel):
    """
    LLAVA + VisionZip: Only does two extra things during initialization:
        1) monkey-patch CLIP's forward methods
        2) wrap the model with visionzip()  
    All other prepare_inputs / generate reuse LlavaModel's implementation.
    """
    def __init__(
        self,
        model_path: str,
        dominant: int = 54,
        contextual: int = 10,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
    ):
        # ---- monkey patch CLIP ----
        self._apply_visionzip_monkey_patches()

        super().__init__(
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        from visionzip import visionzip as _visionzip
        self.model = _visionzip(self.model, dominant=dominant, contextual=contextual)
        self.model.eval()

    @staticmethod
    def _apply_visionzip_monkey_patches():
        """
        make sure CLIP's encoder layer / attention forward methods are patched by VisionZip versions.
        This function is designed to be idempotent: multiple calls will only apply the patch once.
        """
        from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer
        from visionzip.utils import CLIPAttention_forward, CLIP_EncoderLayer_forward

        if getattr(CLIPAttention, "_visionzip_patched", False):
            return

        CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
        CLIPAttention.forward = CLIPAttention_forward

        CLIPAttention._visionzip_patched = True

class LlavaVisionZipModelHybridAttn(LlavaVisionZipModel):
    def __init__(self, model_path: str, dominant: int=54, contextual: int=10,
                 temperature: float=0.2, max_new_tokens: int=256,
                 alpha_config: Tuple[float, float, float] = (1.2, 0.9, 0.2)):

        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates

        # ---- monkey patch CLIP ----
        from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer
        from visionzip.utils import CLIPAttention_forward, CLIP_EncoderLayer_forward
        CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
        CLIPAttention.forward = CLIPAttention_forward

        model_name = get_model_name_from_path(model_path)

        # extra_kwargs = {
        #     "device_map": "auto",
        #     "torch_dtype": torch.float16,
        # }

        # loaded = False
        # last_err: Exception | None = None
        # tokenizer = model = image_processor = None

        # # try 4bit -> 8bit -> fp16
        # attempts = [
        #     ({"load_4bit": True,
        #       "bnb_4bit_compute_dtype": torch.float16,
        #       "bnb_4bit_quant_type": "nf4"}, "4bit"),
        #     ({"load_8bit": True}, "8bit"),
        #     ({}, "fp16"),
        # ]

        # for k, name in attempts:
        #     try:
        #         kwargs = extra_kwargs | k
        #         print(f"[load] trying {name} ...")
        #         tokenizer, model, image_processor, _ = load_pretrained_model(
        #             model_path=model_path,
        #             model_base=None,
        #             model_name=model_name,
        #             **kwargs,
        #         )
        #         print(f"[load] success with {name}")
        #         loaded = True
        #         break
        #     except Exception as e:
        #         last_err = e
        #         print(f"[load] try {name} failed:", e)
        #         # clear memory before next attempt
        #         gc.collect()
        #         if torch.cuda.is_available():
        #             torch.cuda.empty_cache()
        #         continue

        # if not loaded or model is None:
        #     raise RuntimeError(
        #         f"Failed to load LLaVA model in 4bit/8bit/fp16. Last error: {last_err}"
        #     )
        
        print("[load] using pure fp16 on single GPU ...")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device_map=None,  
            torch_dtype=torch.float16,
        )
        print("[load] fp16 load success")
        # ---- move whole model to GPU ----
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self._device = device
        print(f"[load] model moved to {self._device}")

        # ---- VisionZip wrapper ----
        from visionzip import visionzip as _visionzip
        model = _visionzip(model, dominant=dominant, contextual=contextual)

        # (Optional) patch VisionTower EXP
        try:
            from utils.clip_encoder_exp import CLIPVisionTower_VisionZip_EXP_HybridAttn
            from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
            

            vision_tower = getattr(model, "model", None)
            vision_tower = getattr(vision_tower, "vision_tower", None)
            
            print("alpha_config here: ", alpha_config)

            if vision_tower is not None:
                vision_tower._vision_zip_alpha = alpha_config
                print(f"[config] Alpha set on Vision Tower: {alpha_config}")

            CLIPVisionTower.forward = CLIPVisionTower_VisionZip_EXP_HybridAttn.forward
        except Exception as e:
            print("[Warn] EXP forward not found or failed to patch:", e)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_templates = conv_templates
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer_image_token = self._make_tokenizer_image_token()
        self.model.eval()


class LlavaVisionZipModelDynamicK(LlavaVisionZipModel):
    def __init__(self, model_path: str, dominant: int=54, contextual: int=10,
                 temperature: float=0.2, max_new_tokens: int=256,):

        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates

        # ---- monkey patch CLIP ----
        from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer
        from visionzip.utils import CLIPAttention_forward, CLIP_EncoderLayer_forward
        CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
        CLIPAttention.forward = CLIPAttention_forward

        model_name = get_model_name_from_path(model_path)
        
        print("[load] using pure fp16 on single GPU ...")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device_map=None,  
            torch_dtype=torch.float16,
        )
        print("[load] fp16 load success")
        # ---- move whole model to GPU ----
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self._device = device
        print(f"[load] model moved to {self._device}")

        # ---- VisionZip wrapper ----
        from visionzip import visionzip as _visionzip
        model = _visionzip(model, dominant=dominant, contextual=contextual)

        # (Optional) patch VisionTower EXP
        try:
            from utils.clip_encoder_exp import CLIPVisionTower_VisionZip_EXP_DynamicK
            from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

            CLIPVisionTower.forward = CLIPVisionTower_VisionZip_EXP_DynamicK.forward
        except Exception as e:
            print("[Warn] EXP forward not found or failed to patch:", e)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_templates = conv_templates
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer_image_token = self._make_tokenizer_image_token()
        self.model.eval()
    


class SparseVLMModel(LlavaModel):
    """
    remmember to change to env with sparsevlm's llava
    """
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        retained_tokens: int = 192,
        **unused_kwargs,
    ):
        super().__init__(
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        # self.retained_tokens = int(retained_tokens)
        # print(f"[SparseVLMModel] retained_tokens set to {self.retained_tokens}")

# =============================== #
#   LLaVA + VisionZip + SparseZip
#   (hybrid scoring + dyn-K + ctx merge)
# =============================== #
class LlavaSparseZipModel(LlavaModel):
    """
    LLaVA + VisionZip + SparseZip
    """
    def __init__(
        self,
        model_path: str,
        dominant: int = 54,
        contextual: int = 16,
        alpha_config: Tuple[float, float, float] = (1.2, 0.9, 0.2),
        tau_feat: float = 0.2,
        tau_sim: float = 0.1,
        cross_beta: float = 0.0,
        dynamic_k: bool = True,
        dynk_c: float = 8.0,
        k_min: int = 4,
        k_max: int = 64,
        contextual_num: Optional[int] = None,
        kmeans_init_factor: float = 2.0,
        kmeans_iters: int = 10,
        agglomerative: bool = True,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        skip_hybrid_attn: bool = False,
        skip_dynamic_k: bool = False,
        skip_ctx_merge: bool = False,
        sparsezip_cfg: Optional[Dict[str, Any]] = None,
    ):
        from utils.sparsezip_compressor import (
            ScoringAlphas, DynamicKConfig, MergingConfig,
            CompressionConfig, VisionZipCompressor,
        )

        cfg = sparsezip_cfg or {}

        skip_hybrid_attn = bool(cfg.get("skip_hybrid_attn", skip_hybrid_attn))
        skip_dynamic_k   = bool(cfg.get("skip_dynamic_k",   skip_dynamic_k))
        skip_ctx_merge   = bool(cfg.get("skip_ctx_merge",   skip_ctx_merge))

        dynamic_k = bool(cfg.get("dynamic_k", dynamic_k))
        k_min     = int(cfg.get("k_min", k_min))
        k_max     = int(cfg.get("k_max", k_max))
        dynk_cfg  = cfg.get("dynk", {})
        dynk_c    = float(dynk_cfg.get("c",   dynk_c))
        
        alphas_cfg = cfg.get("alphas", {})
        tau_feat   = float(cfg.get("tau_feat", tau_feat))
        tau_sim    = float(cfg.get("tau_sim",  tau_sim))
        cross_beta = float(cfg.get("cross_beta", cross_beta))

        merging_cfg     = cfg.get("merging", {})
        contextual_num  = merging_cfg.get("contextual_num", contextual_num)
        kmeans_init_factor = float(merging_cfg.get("kmeans_init_factor", kmeans_init_factor))
        kmeans_iters       = int(merging_cfg.get("kmeans_iters",       kmeans_iters))
        agglomerative      = bool(merging_cfg.get("agglomerative",     agglomerative))

        LlavaVisionZipModel._apply_visionzip_monkey_patches()

        super().__init__(
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        from visionzip import visionzip as _visionzip
        self.model = _visionzip(self.model, dominant=dominant, contextual=contextual)
        self.model.eval()

        vt = getattr(self.model, "model", None)
        vt = getattr(vt, "vision_tower", None)
        if vt is None:
            print("[SparseZip] vision tower not found, abort.")
            return

        if skip_hybrid_attn:
            alphas = ScoringAlphas(attn=1.0, entropy=0.0, mutual=0.0)
        else:
            alphas = ScoringAlphas(
                attn=float(alphas_cfg.get("attn",    alpha_config[0])),
                entropy=float(alphas_cfg.get("entropy", alpha_config[1])),
                mutual=float(alphas_cfg.get("mutual",  alpha_config[2])),
            )

        k_min_eff = k_min
        k_max_eff = k_max
        use_dynamic_k = dynamic_k and (not skip_dynamic_k)

        k_min_eff = min(k_min_eff, dominant)
        k_max_eff = max(k_max_eff, dominant)

        if not use_dynamic_k:
            k_min_eff = k_max_eff = dominant  

        dynk = DynamicKConfig(
            c=dynk_c,
            k_min=k_min_eff,
            k_max=k_max_eff,
        )

        ctx_num = contextual_num if contextual_num is not None else contextual
        merging = MergingConfig(
            contextual_num=ctx_num,
            kmeans_init_factor=kmeans_init_factor,
            kmeans_iters=kmeans_iters,
            agglomerative=agglomerative,
        )

        comp_cfg = CompressionConfig(
            alphas=alphas,
            tau_feat=tau_feat,
            tau_sim=tau_sim,
            cross_beta=cross_beta,
            dynamic_k=use_dynamic_k,
            dynk=dynk,
            k_min=k_min_eff,
            k_max=k_max_eff,
            merging=merging,
            skip_ctx_merge=skip_ctx_merge,
        )

        vz_comp = VisionZipCompressor(
            num_scoring_layers=1,
            cfg=comp_cfg,
        )

        vt._vz_comp = vz_comp
        self._sparsezip_cfg = comp_cfg

        from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
        from utils.clip_encoder_sparsezip_exp import CLIPVisionTower_SparseZip_EXP
        CLIPVisionTower.forward = CLIPVisionTower_SparseZip_EXP.forward

        print("[SparseZip] VisionZipCompressor attached. "
              f"ctx={ctx_num}, k_range=[{k_min_eff},{k_max_eff}], "
              f"alphas=({alphas.attn},{alphas.entropy},{alphas.mutual}), "
              f"dynamic_k={use_dynamic_k}, skip_ctx_merge={skip_ctx_merge}, "
              f"skip_hybrid_attn={skip_hybrid_attn}, skip_dynamic_k={skip_dynamic_k}")


class LlavaVisionZipModelTextAware(BaseModel):
    """
    LLaVA + VisionZip + Text-aware encoder-side token selection
    """

    def __init__(
        self,
        model_path: str,
        dominant: int = 54,
        contextual: int = 10,
        alpha: float = 0.5,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
    ):
        from transformers import CLIPModel, AutoTokenizer
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates

        # ---- 1. monkey patch CLIPEncoderLayer / CLIPAttention（VisionZip） ----
        from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer
        from visionzip.utils import CLIPAttention_forward, CLIP_EncoderLayer_forward

        CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
        CLIPAttention.forward = CLIPAttention_forward

        model_name = get_model_name_from_path(model_path)
        print("[load] using pure fp16 on single GPU ...")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device_map=None,
            torch_dtype=torch.float16,
        )
        print("[load] fp16 load success")

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model.to(device)
        self._device = device
        print(f"[load] model moved to {self._device}")

        from visionzip import visionzip as _visionzip

        model = _visionzip(model, dominant=dominant, contextual=contextual)

        try:
            from utils.clip_encoder_textaware import CLIPVisionTower_VisionZip_TextAware
            from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

            CLIPVisionTower.forward = CLIPVisionTower_VisionZip_TextAware.forward

            vision_tower = model.get_vision_tower()

            setattr(vision_tower, "visionzip_dominant", int(dominant))
            setattr(vision_tower, "visionzip_contextual", int(contextual))
            setattr(vision_tower, "visionzip_alpha", float(alpha))

            vt_name = getattr(vision_tower, "vision_tower_name", None)
            if vt_name is None:
                vt_cfg = getattr(vision_tower, "vision_tower_cfg", None)
                if vt_cfg is not None and hasattr(vt_cfg, "vision_tower"):
                    vt_name = vt_cfg.vision_tower

            if vt_name is None:
                raise ValueError("Cannot determine CLIP vision tower name to load text encoder")

            self.clip_text_tokenizer = AutoTokenizer.from_pretrained(vt_name)
            clip_model = CLIPModel.from_pretrained(
                vt_name,
                torch_dtype=torch.float16,
            ).to(self._device)
            self.clip_text_model = clip_model.text_model
            self.clip_text_model.eval()

        except Exception as e:
            print("[Warn] Text-aware VisionZip forward patch failed:", e)
            self.clip_text_model = None
            self.clip_text_tokenizer = None

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_templates = conv_templates
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer_image_token = self._make_tokenizer_image_token()
        self.model.eval()