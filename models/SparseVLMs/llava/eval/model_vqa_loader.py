import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import logging
import datetime
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # Filter out questions with missing image files
    original_count = len(questions)
    available_questions = []
    missing_count = 0
    for q in questions:
        image_file = q.get("image", "")
        image_path = os.path.join(args.image_folder, image_file)
        if os.path.exists(image_path):
            available_questions.append(q)
        else:
            missing_count += 1
            # Only print individual warnings if there are few missing files (less than 10)
            if missing_count <= 10:
                print(f"⚠ Skipping question {q.get('question_id', 'unknown')}: image file not found: {image_path}")
    
    if len(available_questions) < original_count:
        print(f"⚠ Filtered {missing_count} questions with missing images (out of {original_count} total)")
        print(f"   Processing {len(available_questions)} questions with available images")
    
    questions = available_questions
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode)
    
    retained_tokens = args.retained_tokens
    # Determine device from model (in case it's on CPU)
    # Try to get device from model parameters - handle multi-device case
    try:
        model_device = next(model.parameters()).device
        # If model_device is meta device, try to get from a different attribute
        if str(model_device) == 'meta':
            # Try to get device from model's dtype or other attributes
            if hasattr(model, 'device'):
                model_device = model.device
            else:
                # Check if any named parameter has a real device
                for name, param in model.named_parameters():
                    if param.device.type != 'meta':
                        model_device = param.device
                        break
                else:
                    model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except (StopIteration, AttributeError):
        # Fallback: check if CUDA is available
        model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"   Model device detected: {model_device}")
    import gc
    for idx_count, ((input_ids, image_tensor, image_sizes), line) in enumerate(tqdm(zip(data_loader, questions), total=len(questions))):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device=model_device, non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device=model_device, non_blocking=True)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                retained_tokens = retained_tokens,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Explicitly delete intermediate tensors
        del input_ids, image_tensor, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
        
        # Clear GPU cache more aggressively to prevent memory accumulation
        if (idx_count + 1) % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                # Force synchronization to ensure all operations complete
                torch.cuda.synchronize()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--retained_tokens", type=int, default=192)
    args = parser.parse_args()

    eval_model(args)
