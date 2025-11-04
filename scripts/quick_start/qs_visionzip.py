import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'VisionZip')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'LLaVA')))


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from PIL import Image
from llava.conversation import conv_templates, SeparatorStyle, Conversation
import torch
from visionzip import visionzip

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
## VisoinZip retains 54 dominant tokens and 10 contextual tokens
model = visionzip(model, dominant=54, contextual=10)

# ------------------ 调试代码 ------------------
# # 1) 检查外层 wrapper 的 forward 是否已经被 VisionZip 版本替换
# from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower as CLIPVT_ORIG
# from visionzip.clip_encoder import CLIPVisionTower_VisionZip

# vt = model.get_model().get_vision_tower()

# print("[Check] vt class:", type(vt))
# print("[Check] vt.forward is patched:",
#       vt.forward.__code__ is CLIPVisionTower_VisionZip.forward.__code__)

# # 2) 检查注意力与EncoderLayer是否已被 monkey-patch
# from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer
# from visionzip.utils import CLIPAttention_forward as VZ_AttnFwd, CLIP_EncoderLayer_forward as VZ_EncFwd

# print("[Check] CLIPAttention.forward patched:",
#       CLIPAttention.forward.__code__ is VZ_AttnFwd.__code__)
# print("[Check] CLIPEncoderLayer.forward patched:",
#       CLIPEncoderLayer.forward.__code__ is VZ_EncFwd.__code__)

# # 3) 检查内层 HF 视觉主干是否换成 VisionZipTransformer，并确认 r 调度
# inner_vit = vt.vision_tower  # 这是真正的 HF CLIPVisionModel/Transformer 容器
# print("[Check] inner vision class:", inner_vit.__class__)
# print("[Check] r schedule:", getattr(inner_vit, "r", None))
# ----------------------------------------------
# CLIPVisionTower_VisionZip替换检查
# import types, inspect
# import llava.model.multimodal_encoder.clip_encoder as clip_mod
# from visionzip.clip_encoder import CLIPVisionTower_VisionZip

# vt = model.get_model().get_vision_tower()

# # 1) 确认我们和实例使用的是同一个类对象
# print("[Path]", clip_mod.__file__)
# print("[Same Class]", vt.__class__ is clip_mod.CLIPVisionTower)

# # 2) 先对“类”打补丁（影响之后新建的实例）
# clip_mod.CLIPVisionTower.forward = CLIPVisionTower_VisionZip.forward

# # 3) 再对“现有实例”强制绑定（避免实例级 forward 覆盖）
# vt.forward = types.MethodType(CLIPVisionTower_VisionZip.forward, vt)

# # 4) 复检（注意：某些情况下 vt.forward 可能是实例上的函数，没有 __func__；
# #    用 getattr 兼容两种情况）
# patched = getattr(vt.forward, "__func__", vt.forward).__code__ is CLIPVisionTower_VisionZip.forward.__code__
# print("[Check] vt.forward is patched:", patched)

# # 可选：打印源码开头几行，肉眼确认
# print("[Head of vt.forward]:")
# print("\n".join(inspect.getsource(getattr(vt.forward, "__func__", vt.forward)).splitlines()[:5]))
# ----------------------------
## Inference
image_file = "/u/q/i/qinxinghao/project/VisionZip-exp/reference/owl.JPEG"
prompt = "Describe the image in detail"

image = Image.open(image_file).convert('RGB')
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half()
image_tensor = image_tensor.to(model.device)

conv = conv_templates["llava_v1"].copy()
image_placeholder = DEFAULT_IMAGE_TOKEN # '<image>'

if model.config.mm_use_im_start_end:
    # Like: <s>[INST] <<SYS>>...<</SYS>>\n[INST] <im_start><image><im_end>\nDescribe the image [/INST] ASSISTANT:
    image_placeholder = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

full_user_input = image_placeholder + '\n' + prompt
conv.append_message(conv.roles[0], full_user_input)
conv.append_message(conv.roles[1], None)
final_prompt_string = conv.get_prompt()

input_ids = tokenizer_image_token(
    final_prompt_string, 
    tokenizer, 
    IMAGE_TOKEN_INDEX, 
    return_tensors='pt'
).unsqueeze(0).to(model.device)

# ----------------------------
# GPU 与设备/精度自检
import os, torch
print("cuda.is_available:", torch.cuda.is_available())
print("cuda.device_count:", torch.cuda.device_count())
print("torch.version.cuda:", torch.version.cuda)
print("cudnn.version:", torch.backends.cudnn.version())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

vt = model.get_model().get_vision_tower()
p = next(vt.vision_tower.parameters())
print("vt.device:", vt.device, "vt.dtype:", vt.dtype)
print("vit.weight.device:", p.device, "vit.weight.dtype:", p.dtype)
print("image_tensor.device:", image_tensor.device, "image_tensor.dtype:", image_tensor.dtype)
# ----------------------------
# 先用“最保守路径”验证（排除混精度/驱动小坑）
import torch, os
torch.cuda.init()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # 让报错栈更准确
torch.backends.cudnn.enabled = False       # 临时绕开 cuDNN（只为定位问题）
# 统一 inputs 的 device/dtype 到 vision 权重
vt  = model.get_model().get_vision_tower()
vit = vt.vision_tower

# 用“已加载权重”的设备/精度作为单一真值来源
param = next(vit.parameters())
vt.device = param.device            # e.g., cuda:0
vt.dtype  = param.dtype             # e.g., torch.float32 或 bfloat16/float16

# 重新构造 image_tensor：严格对齐到 vision 的 device/dtype
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
image_tensor = image_tensor.to(vt.device, dtype=vt.dtype)

# 保险起见，关掉 cuDNN 的 benchmark（避免首次 conv2d 初始化的一些坑）
import torch
torch.backends.cudnn.benchmark = False

# 干跑一遍 CLIP vision（不移动模块，只喂同 dtype/device 的输入）
with torch.inference_mode():
    _ = vit(image_tensor, output_hidden_states=True, output_attentions=True)
print("[OK] CLIP vision forward passed")

# 再跑 generate（VisionZip 会自动在 forward 里把 images.to(vt.device, vt.dtype)）
# ----------------------------

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=512,
        use_cache=True,
    )

output = tokenizer.decode(output_ids[0]).strip()
print("Model response:", output)
