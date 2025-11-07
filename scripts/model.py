from typing import Any, Dict
import time
from PIL import Image
import torch
from utils.timer import StageTimer
from scripts.abstract import BaseModel, Sample
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates

# ========================= #
# Model: LLaVA + VisionZip  #
# ========================= #

class LlavaVisionZipModel(BaseModel):
    """
    Wrap your existing single-sample pipeline; keep it deterministic & timed.
    """
    def __init__(self, model_path: str, dominant: int=54, contextual: int=10, temperature: float=0.2, max_new_tokens: int=256):
        
        # ---- monkey patches (must be before loading model) ----
        from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer
        from visionzip.utils import CLIPAttention_forward, CLIP_EncoderLayer_forward
        CLIPAttention.forward = CLIPAttention_forward
        CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward

        # (Optional) override CLIPVisionTower.forward if you have a custom EXP version
        try:
            from utils.clip_encoder_exp import CLIPVisionTower_VisionZip_EXP
            from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
            CLIPVisionTower.forward = CLIPVisionTower_VisionZip_EXP.forward
        except Exception as e:
            print("[Warn] EXP forward not found or failed to patch:", e)

        # ---- load model ----
        extra_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        # prefer 4bit, else 8bit, else fp16
        for k, v in [
            ({"load_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4"}, "4bit"),
            ({"load_8bit": True}, "8bit"),
            ({}, "fp16")
        ]:
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
        
        # apply VisionZip wrapper
        from visionzip import visionzip as _visionzip
        model = _visionzip(model, dominant=dominant, contextual=contextual)

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

        self.model.eval()

    def device(self) -> torch.device:
        return self.model.device

    @torch.inference_mode()
    def prepare_inputs(self, sample: Sample) -> Dict[str, Any]:
        """prepare inputs for a single sample
        Returns dict with keys:
          - 'input_ids': torch.LongTensor
          - 'images': torch.FloatTensor or None
          - 'load_ms': float
          - 'preprocess_ms': float
        """
        # image
        image_tensor = None
        t0 = time.perf_counter()
        img = Image.open(sample.image_path).convert("RGB") if sample.image_path else None
        load_ms = (time.perf_counter() - t0)*1000.0

        t1 = time.perf_counter()
        if img is not None:
            image_tensor = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'].half()
            image_tensor = image_tensor.to(self.device())
        preprocess_ms = (time.perf_counter() - t1)*1000.0

        # prompt
        conv = self.conv_templates["llava_v1"].copy()
        image_placeholder = self.DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_placeholder = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
        full_user_input = image_placeholder + "\n" + sample.prompt
        conv.append_message(conv.roles[0], full_user_input)
        conv.append_message(conv.roles[1], None)
        final_prompt_string = conv.get_prompt()

        input_ids = tokenizer_image_token(
            final_prompt_string, 
            self.tokenizer, 
            self.IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).to(self.device())

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "load_ms": load_ms,
            "preprocess_ms": preprocess_ms,
        }

    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        timer = StageTimer(use_cuda=(self.device().type == "cuda"))

        # === 计时（粗粒度） ===
        timer.start("end2end")
        timer.start("decode")  # 单次 generate 视作 decode 段（prefill 先置 0）

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
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        # —— 兼容“拼接返回/仅生成返回”的新 token 数 —— 
        try:
            in_len  = inputs["input_ids"].shape[1]
            seq_len = out_ids.shape[1]
            gen_len = int(seq_len - in_len) if seq_len >= in_len else int(seq_len)
        except Exception:
            gen_len = len(self.tokenizer(text, add_special_tokens=False).input_ids)

        timings_ms = {
            "load_ms":       inputs.get("load_ms", 0.0),
            "preprocess_ms": inputs.get("preprocess_ms", 0.0),
            "prefill_ms":    0.0,  # 粗粒度先不拆
            "decode_ms":     timer.result_ms("decode"),
            "end2end_ms":    timer.result_ms("end2end"),
            "num_new_tokens": int(max(gen_len, 0)),
        }
        return {"text": text, **timings_ms}