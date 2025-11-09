from typing import Any, Dict, Tuple, List, Optional, Union
import time
from PIL import Image
import torch
from scripts.timer import StageTimer
from scripts.abstract import BaseModel, Sample

# ========================= #
# Model: LLaVA + VisionZip  #
# ========================= #

class LlavaVisionZipModel(BaseModel):
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

        # ---- Load model ----
        import platform
        extra_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        quant_attempts: List[Tuple[Dict[str, Any], str]]
        if platform.system() == "Darwin":
            # bitsandbytes not supported on macOS arm; skip 4bit/8bit attempts
            quant_attempts = [({}, "fp16")]
        else:
            quant_attempts = [
                ({"load_4bit": True,
                  "bnb_4bit_compute_dtype": torch.float16,
                  "bnb_4bit_quant_type": "nf4"}, "4bit"),
                ({"load_8bit": True}, "8bit"),
                ({}, "fp16"),
            ]
        for k, v in quant_attempts:
            try:
                kwargs = extra_kwargs | k
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=get_model_name_from_path(model_path),
                    **kwargs
                )
                print(f"[load] success with {v}")
                break
            except Exception as e:
                print(f"[load] try {v} failed:", e)
                continue
        
        # Apply VisionZip wrapper to the model
        from visionzip import visionzip as _visionzip
        model = _visionzip(model, dominant=dominant, contextual=contextual)
        
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


class SparseVLMModel(BaseModel):
    """
    A single-sample wrapper for SparseVLM-style LLaVA models.
    - Deterministic preprocessing
    - Precise timing
    - Compatible with your existing eval pipeline
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        retained_tokens: int = 192,
        conv_mode: str = "llava_v1",
    ):
        # ---- imports kept inside to avoid heavy import on module import ----
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates

        # ---- load model (4bit -> 8bit -> fp16) ----
        extra_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        for k, tag in [
            ({"load_4bit": True, "bnb_4bit_compute_dtype": torch.float16, "bnb_4bit_quant_type": "nf4"}, "4bit"),
            ({"load_8bit": True}, "8bit"),
            ({}, "fp16"),
        ]:
            try:
                kwargs = extra_kwargs | k
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=get_model_name_from_path(model_path),
                    **kwargs,
                )
                print(f"[load] success with {tag}")
                break
            except Exception as e:
                print(f"[load] try {tag} failed:", e)
                continue

        # ---- cache handles on self ----
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_templates = conv_templates
        self.conv_mode = conv_mode

        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN

        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.retained_tokens = int(retained_tokens)

        # function wrappers with fallbacks
        self.tokenizer_image_token = self._make_tokenizer_image_token()
        self._process_images = self._make_process_images()

        self.model.eval()

    # ------------------------ helpers ------------------------

    def _make_tokenizer_image_token(self):
        """
        Prefer LLaVA's tokenizer_image_token; else fallback:
        - replace <image> with IMAGE_TOKEN_INDEX
        - drop <im_start>/<im_end>
        Always returns torch.LongTensor
        """
        try:
            from llava.mm_utils import tokenizer_image_token as _tik

            def _wrap(prompt: str, **kwargs) -> torch.Tensor:
                # try legacy sig first, then keyword style
                try:
                    out = _tik(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, **kwargs)
                except TypeError:
                    kwargs.setdefault("tokenizer", self.tokenizer)
                    kwargs.setdefault("image_token_index", self.IMAGE_TOKEN_INDEX)
                    out = _tik(prompt, **kwargs)
                if not torch.is_tensor(out):
                    out = torch.tensor(out, dtype=torch.long)
                return out

            return _wrap
        except Exception:
            print("[warn] tokenizer_image_token not found; using local fallback.")

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

            def _local(prompt: str, **kwargs) -> torch.Tensor:
                ids = tok(prompt, return_tensors=None, add_special_tokens=True)["input_ids"]
                new_ids = []
                for t in ids:
                    if tok_img is not None and t == tok_img:
                        new_ids.append(self.IMAGE_TOKEN_INDEX)
                    elif tok_im_start is not None and t == tok_im_start:
                        continue
                    elif tok_im_end is not None and t == tok_im_end:
                        continue
                    else:
                        new_ids.append(t)
                return torch.tensor(new_ids, dtype=torch.long)

            return _local

    def _make_process_images(self):
        """
        Prefer LLaVA's process_images(images, image_processor, model_config).
        Fallback to image_processor.preprocess for single image.
        Returns FloatTensor on the model device.
        """
        try:
            from llava.mm_utils import process_images as _pimg

            def _wrap(pil_img: Image.Image) -> Tuple[torch.FloatTensor, List[Tuple[int, int]]]:
                imgs = [pil_img]
                # _pimg returns a batch tensor (B, C, H, W); sizes not returned, so we collect from PIL
                tensor = _pimg(imgs, self.image_processor, self.model.config)[0]
                sizes = [pil_img.size]  # (W, H)
                return tensor, sizes

            return _wrap
        except Exception:
            print("[warn] process_images not found; using processor.preprocess fallback.")

            def _local(pil_img: Image.Image) -> Tuple[torch.FloatTensor, List[Tuple[int, int]]]:
                out = self.image_processor.preprocess(pil_img, return_tensors="pt")["pixel_values"][0]
                sizes = [pil_img.size]
                return out, sizes

            return _local

    # --------------------- BaseModel API ---------------------

    def device(self) -> torch.device:
        return self.model.device

    @torch.inference_mode()
    def prepare_inputs(self, sample: Sample) -> Dict[str, Any]:
        """
        Returns:
          - input_ids: LongTensor [1, L]
          - images:    FloatTensor [1, C, H, W] (half moved to model device)
          - image_sizes: list[(W,H)] length=1
          - load_ms / preprocess_ms
        """
        # image load
        image_tensor = None
        image_sizes = None

        t0 = time.perf_counter()
        img = Image.open(sample.image_path).convert("RGB") if sample.image_path else None
        load_ms = (time.perf_counter() - t0) * 1000.0

        # preprocess
        t1 = time.perf_counter()
        if img is not None:
            img_tensor, sizes = self._process_images(img)
            image_tensor = img_tensor.to(dtype=torch.float16, device=self.device(), non_blocking=True).unsqueeze(0)
            image_sizes = sizes  # list of (W,H)
        preprocess_ms = (time.perf_counter() - t1) * 1000.0

        # prompt compose
        from llava.conversation import conv_templates  # reimport safe
        conv = conv_templates[self.conv_mode].copy()

        # image placeholder
        image_placeholder = self.DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_placeholder = (
                self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
            )

        full_user_input = image_placeholder + "\n" + sample.prompt
        conv.append_message(conv.roles[0], full_user_input)
        conv.append_message(conv.roles[1], None)
        final_prompt_string = conv.get_prompt()

        # tokenize prompt (with image sentinel -> IMAGE_TOKEN_INDEX)
        ids = self.tokenizer_image_token(final_prompt_string, return_tensors=None)
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
        input_ids = ids.unsqueeze(0).to(self.device())

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "image_sizes": image_sizes,
            "load_ms": load_ms,
            "preprocess_ms": preprocess_ms,
        }

    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls model.generate with SparseVLM-style args and returns text + timings.
        """
        timer = StageTimer(use_cuda=(self.device().type == "cuda"))

        timer.start("end2end")
        timer.start("decode")  # treat all as decode in coarse timer

        out = self.model.generate(
            inputs=inputs["input_ids"],
            images=inputs["images"],
            image_sizes=inputs.get("image_sizes", None),
            retained_tokens=self.retained_tokens,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=None, 
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
        )

        timer.end("decode")
        timer.end("end2end")

        out_ids = out.sequences
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        # new tokens
        try:
            in_len = inputs["input_ids"].shape[1]
            seq_len = out_ids.shape[1]
            gen_len = int(seq_len - in_len) if seq_len >= in_len else int(seq_len)
        except Exception:
            gen_len = len(self.tokenizer(text, add_special_tokens=False).input_ids)

        timings_ms = {
            "load_ms": inputs.get("load_ms", 0.0),
            "preprocess_ms": inputs.get("preprocess_ms", 0.0),
            "prefill_ms": 0.0,
            "decode_ms": timer.result_ms("decode"),
            "end2end_ms": timer.result_ms("end2end"),
            "num_new_tokens": int(max(gen_len, 0)),
        }
        return {"text": text, **timings_ms}