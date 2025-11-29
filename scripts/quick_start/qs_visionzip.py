import sys
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..')) 

if project_root not in sys.path:
    sys.path.append(project_root)

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

## Inference
image_file = "/home/w1nd519994824/project/VisionZip-exp/reference/owl.JPEG"
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
