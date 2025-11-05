import os, json, time, math, csv, random, string
from typing import List, Dict, Any, Optional, Iterable
from scripts.dataset import VQAv2Dataset
from scripts.abstract import BaseDataset, BaseModel, BaseMetric, Sample
import torch
from PIL import Image
from utils.timer import StageTimer
from scripts.evaluate import Evaluator
from scripts.metric import ExactMatch, VQASoftAcc, DelayStats
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
        # 优先 4bit；不行就回退 8bit；再不行就 fp16
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
        # tokenizer, model, image_processor, _ = load_pretrained_model(
        #     model_path=model_path, model_base=None, model_name=get_model_name_from_path(model_path)
        # )
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
        # stage timers (CUDA aware)
        timer = StageTimer(use_cuda=(self.device().type == "cuda"))

        # Prefill + decode segmentation: approximate segmentation using a single generate call (coarse-grained)
        # Since HF generate doesn't directly expose prefill/decoding stages, we use the total latency.
        # For finer granularity, we could use `stopping_criteria` and step-by-step decode loops.
        timer.start("end2end")
        timer.start("prefill")  # Approximation: treat the first forward pass as prefill
        # Decide greedy vs sampling based on temperature
        do_sample = (self.temperature is not None and self.temperature > 0.0)

        gen_kwargs = dict(
            inputs=inputs["input_ids"],
            images=inputs["images"],
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )

        if do_sample:
            gen_kwargs.update(dict(do_sample=True, temperature=float(self.temperature)))
        else:
            # Greedy: do_sample=False; do NOT pass temperature
            gen_kwargs.update(dict(do_sample=False))

        out_ids = self.model.generate(**gen_kwargs)
        # We can only precisely get end2end; prefill/decoding use approximate splitting (can be replaced with token-by-token decoding)
        timer.end("prefill")
        timer.end("end2end")

        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        # timings_ms = {
        #     "load_ms": inputs.get("load_ms", 0.0),
        #     "preprocess_ms": inputs.get("preprocess_ms", 0.0),
        #     "prefill_ms": timer.result_ms("prefill"),
        #     # Ensure decode_ms is non-negative
        #     "decode_ms": max(timer.result_ms("end2end") - timer.result_ms("prefill"), 0.0), 
        #     "end2end_ms": timer.result_ms("end2end"),
        # }
        try:
            num_new_tokens = len(self.tokenizer(text, add_special_tokens=False).input_ids)
        except Exception:
            num_new_tokens = 0

        timings_ms = {
            "load_ms": inputs.get("load_ms", 0.0),
            "preprocess_ms": inputs.get("preprocess_ms", 0.0),
            "prefill_ms": timer.result_ms("prefill"),
            "decode_ms": max(timer.result_ms("end2end") - timer.result_ms("prefill"), 0.0),
            "end2end_ms": timer.result_ms("end2end"),
            "num_new_tokens": int(num_new_tokens),
        }
        # Estimate the number of generated tokens (excluding prompt)
        try:
            gen_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
            timings_ms["num_new_tokens"] = int(len(gen_ids))
        except Exception:
            timings_ms["num_new_tokens"] = 0

        return {"text": text, **timings_ms}

# ============ #
# Entry (CLI)  #
# ============ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--ann_path", type=str, required=True, help="VQAv2-like annotation json")
    parser.add_argument("--output_dir", type=str, default="./outputs_vqav2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dominant", type=int, default=54)
    parser.add_argument("--contextual", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="vqa", choices=["vqa", "mme"],
                    help="datatset options")
    args = parser.parse_args()

    model = LlavaVisionZipModel(
        model_path=args.model_path,
        dominant=args.dominant,
        contextual=args.contextual,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
    # dataset = VQAv2Dataset(args.ann_path, limit=args.limit)
    if args.dataset == "vqa":
        from scripts.dataset import VQAv2Dataset
        from scripts.metric import ExactMatch, VQASoftAcc
        dataset = VQAv2Dataset(args.ann_path, limit=args.limit)
        metrics = [ExactMatch(), VQASoftAcc(), DelayStats()]
    else:  # MME
        from scripts.dataset import MMEDataset
        from scripts.metric import MMEAcc, MMEAccPlus
        dataset = MMEDataset(args.ann_path, limit=args.limit)
        metrics = [MMEAcc(), MMEAccPlus(), DelayStats()]
    evaluator = Evaluator(model, dataset, metrics, output_dir=args.output_dir, warmup=args.warmup, seed=args.seed)
    evaluator.run(limit=args.limit)