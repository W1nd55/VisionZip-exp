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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self._device = device
        print(f"[load] model moved to {self._device}")

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
        self.retained_tokens = int(retained_tokens)
        print(f"[SparseVLMModel] retained_tokens set to {self.retained_tokens}")

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