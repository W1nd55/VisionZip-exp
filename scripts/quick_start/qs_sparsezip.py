# tools/qs_sparsezip.py
# remember to `conda activate sparsevlm` (or对应env) before running

import os
import sys
import torch
import yaml
from types import SimpleNamespace # 用于将字典转换为对象以便于访问

# ---------- 把项目根目录加到 sys.path ----------
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 假设项目根目录在 ../..
project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------- 导入我们自己的 Wrapper ----------
# 确保 scripts.model 和 scripts.abstract 路径正确
from scripts.model import LlavaSparseZipModel
from scripts.abstract import Sample

def load_config(config_path):
    """加载并解析 YAML 配置文件"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def main():
    # ===== 1. 加载配置 =====
    config_file = "/home/w1nd519994824/project/VisionZip-exp/config/sparsezip_mme.yaml" # 假设你的 YAML 文件名为 config_mme.yaml
    if not os.path.exists(config_file):
        print(f"[ERROR] Config file '{config_file}' not found. Please create it.")
        sys.exit(1)
        
    config = load_config(config_file)
    model_cfg = config['model']
    sparsezip_cfg = model_cfg['sparsezip']

    # 提取 LlavaSparseZipModel 需要的扁平化参数
    model_path   = model_cfg['model_path']
    temperature  = model_cfg['temperature']
    max_new_tokens = model_cfg['max_new_tokens']
    dominant     = model_cfg['dominant']
    contextual   = model_cfg['contextual']

    # 从 sparsezip 配置中提取参数
    dynk_c       = sparsezip_cfg['dynk']['c']
    k_min        = sparsezip_cfg['k_min']
    k_max        = sparsezip_cfg['k_max']
    
    # 构建 alpha_config (attn, entropy, mutual)
    alphas_cfg = sparsezip_cfg['alphas']
    alpha_config = (alphas_cfg['attn'], alphas_cfg['entropy'], alphas_cfg['mutual'])
    
    # contextual_num
    contextual_num = sparsezip_cfg['merging']['contextual_num']

    # ===== 2. 构造模型（会自动 monkey patch + visionzip） =====
    model = LlavaSparseZipModel(
        model_path   = model_path,
        dominant     = dominant,
        contextual   = contextual,
        alpha_config = alpha_config,
        dynk_c       = dynk_c,
        k_min        = k_min,
        k_max        = k_max,
        contextual_num = contextual_num,
        temperature  = temperature,
        max_new_tokens = max_new_tokens,
        # 其他 sparsezip 参数（如 tau_feat, tau_sim, cross_beta 等）可能需要你的 LlavaSparseZipModel
        # 内部处理或通过 kwargs 传递。这里只传递了构造函数中显式列出的参数。
    )

    # 看一下模型 device
    print("[info] model.device =", model.device())
    print(f"[info] Using model path: {model_path}")

    # ===== 3. 配置输入 =====
    image_file   = "/home/w1nd519994824/project/VisionZip-exp/reference/owl.JPEG"
    prompt       = "Describe the image in detail"
    sample_qid = "qs_test_001"

    # 构造单条 Sample
    sample = Sample(
        image_path = image_file,
        prompt     = prompt,
        qid        = sample_qid
    )

    # ===== 4. 预处理 + 生成 =====
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