# rembemer to set conda env for sparsevlm before running this script
import os
import sys

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..')) 

if project_root not in sys.path:
    sys.path.append(project_root)

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

prompt = "Describe the image in detail"
image_file = "/home/w1nd519994824/project/VisionZip-exp/reference/owl.JPEG"

Args = type("Args", (), {})
args = Args()
args.model_path   = model_path
args.model_base   = None
args.model_name   = get_model_name_from_path(model_path)
args.query        = prompt
args.conv_mode    = None
args.image_file   = image_file
args.sep          = ","
args.temperature  = 0.0
args.top_p        = None
args.num_beams    = 1
args.max_new_tokens = 512

eval_model(args)