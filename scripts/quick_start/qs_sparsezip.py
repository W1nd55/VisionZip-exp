# tools/qs_sparsezip.py
# remember to `conda activate sparsevlm` (or对应env) before running

import os
import sys
import torch

# ---------- 把项目根目录加到 sys.path ----------
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------- 导入我们自己的 Wrapper ----------
from scripts.model import LlavaSparseZipModel
from scripts.abstract import Sample   # 你 evalkit 里已经有的类

def main():
    # ===== 配置 =====
    model_path   = "liuhaotian/llava-v1.5-7b"
    image_file   = "/home/w1nd519994824/project/VisionZip-exp/reference/owl.JPEG"
    prompt       = "Describe the image in detail"

    # SparseZip / VisionZip 相关超参（可以先用论文里的默认）
    dominant     = 54          # VisionZip 内部会 dominant-1 当成 patch 数
    contextual   = 10
    alpha_config = (1.2, 0.9, 0.2)
    dynk_c       = 8.0

    # ===== 1. 构建模型（会自动 monkey patch + visionzip） =====
    model = LlavaSparseZipModel(
        model_path   = model_path,
        dominant     = dominant,
        contextual   = contextual,
        alpha_config = alpha_config,
        dynk_c       = dynk_c,
        k_min        = 4,
        k_max        = 64,
        contextual_num = 30,     # 不写就默认等于 contextual
        temperature  = 0.0,
        max_new_tokens = 512,
    )

    # 看一下模型 device
    print("[info] model.device =", model.device())

    # ===== 2. 构造单条 Sample =====
    sample = Sample(
        image_path = image_file,
        prompt     = prompt,
        # 你的 Sample 如果还有其它字段（qid 等）这里一起填上
    )

    # ===== 3. 预处理 + 生成 =====
    inputs = model.prepare_inputs(sample)  # 返回 {"input_ids":..., "pixel_values":..., ...}
    out    = model.generate(inputs)        # 返回 {"text": ..., "load_ms":..., ...}

    print("\n========== Answer ==========")
    print(out["text"])
    print("================================\n")

    print("Timings (ms):")
    for k in ["load_ms", "preprocess_ms", "end2end_ms", "num_new_tokens"]:
        if k in out:
            print(f"  {k}: {out[k]}")

if __name__ == "__main__":
    main()