"""
Evaluate SparseVLM on POPE dataset.
This script loads the model and runs evaluation with SparseVLM sparsification.
"""
import sys
import os
import argparse
import json
import random
import tempfile
import shutil

# Note: transformers package has been patched to skip tokenizers version check
# See venv/Lib/site-packages/transformers/dependency_versions_check.py

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'SparseVLMs')))

import torch
from llava.model.builder import load_pretrained_model  # type: ignore
from llava.mm_utils import get_model_name_from_path  # type: ignore
from llava.eval.model_vqa_loader import eval_model  # type: ignore
import argparse as arg_parse

def main():
    parser = arg_parse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/eval/pope/val2014")
    parser.add_argument("--question-file", type=str, default="./playground/eval/pope/llava_pope_test.jsonl")
    parser.add_argument("--answers-file", type=str, default="./playground/eval/pope/answers/sparsevlm-v1.5-7b.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--retained_tokens", type=int, default=64, help="Number of tokens to retain (64, 128, or 192)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit to N random samples (for quick testing)")
    
    args = parser.parse_args()
    
    # Change to SparseVLMs directory first (paths are relative to this)
    sparsevlm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'SparseVLMs'))
    original_cwd = os.getcwd()
    os.chdir(sparsevlm_dir)
    
    # Resolve paths - they should point to LLaVA's playground directory, not SparseVLMs
    # Question file and image folder are in LLaVA, not SparseVLMs
    llava_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'LLaVA'))
    
    # Resolve paths relative to LLaVA directory (where the data actually is)
    if not os.path.isabs(args.question_file):
        # If relative, assume it's relative to LLaVA directory
        args.question_file = os.path.join(llava_dir, args.question_file.lstrip('./'))
    if not os.path.isabs(args.image_folder):
        args.image_folder = os.path.join(llava_dir, args.image_folder.lstrip('./'))
    if not os.path.isabs(args.answers_file):
        args.answers_file = os.path.join(llava_dir, args.answers_file.lstrip('./'))
    
    # Make all paths absolute
    args.question_file = os.path.abspath(args.question_file)
    args.image_folder = os.path.abspath(args.image_folder)
    args.answers_file = os.path.abspath(args.answers_file)
    
    # If max_samples is set, sample random questions
    if args.max_samples and args.max_samples > 0:
        print(f"\n⚠ Sampling {args.max_samples} random questions from dataset...")
        # Read all questions
        with open(args.question_file, 'r') as f:
            all_questions = [json.loads(line) for line in f]
        
        # Filter out questions with missing images
        available_questions = []
        for q in all_questions:
            image_path = os.path.join(args.image_folder, q.get('image', ''))
            if os.path.exists(image_path):
                available_questions.append(q)
        
        if len(available_questions) < args.max_samples:
            print(f"   ⚠ Only {len(available_questions)} questions have available images (requested {args.max_samples})")
            print(f"   Using all {len(available_questions)} available questions")
            sampled_questions = available_questions
        elif len(available_questions) > args.max_samples:
            sampled_questions = random.sample(available_questions, args.max_samples)
            print(f"   Selected {args.max_samples} random samples from {len(available_questions)} available (out of {len(all_questions)} total)")
        else:
            sampled_questions = available_questions
            print(f"   Using all {len(available_questions)} available questions")
        
        # Create temporary file with sampled questions
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        for q in sampled_questions:
            temp_file.write(json.dumps(q) + '\n')
        temp_file.close()
        
        # Update question file path (save original for later)
        original_question_file = args.question_file
        args.question_file = temp_file.name
        print(f"   Using temporary file: {args.question_file}")
    else:
        temp_file = None
        original_question_file = None
    
    # Paths are already resolved relative to SparseVLMs directory (we're already there)
    
    print("=" * 60)
    print("SparseVLM POPE Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"SparseVLM: retained_tokens={args.retained_tokens}")
    print(f"Question file: {args.question_file}")
    print(f"Image folder: {args.image_folder}")
    print(f"Answers file: {args.answers_file}")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    print("   (This may take a few minutes on first run)")
    try:
        model_name = get_model_name_from_path(args.model_path)
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"   Found {num_gpus} GPU(s):")
            for i in range(num_gpus):
                gpu_mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_mem_alloc = torch.cuda.memory_allocated(i) / (1024**3)
                gpu_mem_cached = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"     GPU {i}: {torch.cuda.get_device_name(i)} - Total: {gpu_mem_total:.1f}GB, Allocated: {gpu_mem_alloc:.2f}GB, Cached: {gpu_mem_cached:.2f}GB")
            
            # Calculate max_memory for 4-bit (leave ~1GB for activations)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_memory_4bit = {0: f"{max(8.0, gpu_mem_total - 1.0):.1f}GiB"}  # Leave 1GB for 4-bit
            print(f"   Setting max_memory for 4-bit to {max_memory_4bit} (leaving ~1GB for activations)")
            
            print("   Attempting to load with 4-bit quantization (lower memory)...")
            try:
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path=args.model_path,
                    model_base=args.model_base,
                    model_name=model_name,
                    load_4bit=True,
                    device_map="auto",
                    max_memory=max_memory_4bit
                )
            except Exception as e:
                print(f"   4-bit loading failed: {e}")
                print("   Falling back to GPU without quantization (float16)...")
                try:
                    # Clear GPU cache first to ensure clean state
                    print("   Clearing GPU cache...")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    # For float16, don't set max_memory - let accelerate maximize GPU usage
                    # A 7B model in float16 needs ~14GB, so it will use device_map="auto"
                    # to offload some layers, but will try to use as much GPU as possible
                    print("   Note: Not setting max_memory - will maximize GPU usage")
                    print("   Model will use device_map='auto' to balance GPU/CPU placement")
                    tokenizer, model, image_processor, context_len = load_pretrained_model(
                        model_path=args.model_path,
                        model_base=args.model_base,
                        model_name=model_name,
                        device_map="auto",  # Allow automatic device placement with CPU offloading
                        device="cuda"  # Explicitly set device for flash attention layers
                        # No max_memory - let accelerate use as much GPU as possible
                    )
                except Exception as e2:
                    print(f"   GPU loading (float16) also failed: {e2}")
                    print("   Falling back to CPU...")
                    tokenizer, model, image_processor, context_len = load_pretrained_model(
                        model_path=args.model_path,
                        model_base=args.model_base,
                        model_name=model_name,
                        device_map="cpu",
                        device="cpu"
                    )
        else:
            print("   Using CPU (will be slower)")
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=args.model_path,
                model_base=args.model_base,
                model_name=model_name,
                device_map="cpu",
                device="cpu"  # Explicitly set device for flash attention layers
            )
        print("   ✓ Model loaded")
        
        # Diagnose model device placement
        if torch.cuda.is_available():
            print("\n   Checking model device placement...")
            gpu_params = 0
            cpu_params = 0
            total_params = 0
            for name, param in model.named_parameters():
                total_params += param.numel()
                if param.device.type == 'cuda':
                    gpu_params += param.numel()
                else:
                    cpu_params += param.numel()
            
            gpu_pct = (gpu_params / total_params * 100) if total_params > 0 else 0
            cpu_pct = (cpu_params / total_params * 100) if total_params > 0 else 0
            
            print(f"   GPU parameters: {gpu_params/1e9:.2f}B ({gpu_pct:.1f}%)")
            print(f"   CPU parameters: {cpu_params/1e9:.2f}B ({cpu_pct:.1f}%)")
            
            if gpu_pct < 50:
                print("   ⚠ WARNING: Most model is on CPU - this will be very slow!")
                print("   Consider removing max_memory limit to use more GPU")
            else:
                print("   ✓ Most model is on GPU")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Monkey-patch the eval_model function to use our model
    print("\n2. Running evaluation...")
    import llava.eval.model_vqa_loader as model_vqa_module  # type: ignore
    
    # Save original function
    original_load = model_vqa_module.load_pretrained_model
    
    # Create a closure to capture our model
    def patched_load(model_path, model_base, model_name):
        return tokenizer, model, image_processor, context_len
    
    # Replace the function
    model_vqa_module.load_pretrained_model = patched_load
    
    # Determine device from model and patch .cuda() calls
    model_device = next(model.parameters()).device
    print(f"   Using device: {model_device}")
    
    # Check if model is actually on CPU (even if CUDA is available)
    model_is_cpu = str(model_device) == 'cpu'
    
    # Monkey-patch tensor.cuda() to use model's device
    original_tensor_cuda = torch.Tensor.cuda
    
    def device_aware_cuda(self, device=None):
        """Replace .cuda() to use model's device"""
        if model_is_cpu:
            # If model is on CPU, always use CPU
            return self.to(model_device)
        elif torch.cuda.is_available() and device is None:
            return original_tensor_cuda(self)
        else:
            return self.to(model_device)
    
    torch.Tensor.cuda = device_aware_cuda
    
    # Also patch the .to(device='cuda') pattern used in model_vqa_loader
    original_tensor_to = torch.Tensor.to
    
    def device_aware_to(self, *args, **kwargs):
        """Replace .to(device='cuda') with model's device"""
        # Check if trying to move to CUDA but model is on CPU
        if model_is_cpu:
            if 'device' in kwargs and kwargs['device'] == 'cuda':
                kwargs['device'] = model_device
            elif len(args) > 0 and args[0] == 'cuda':
                args = (model_device,) + args[1:]
        elif 'device' in kwargs and kwargs['device'] == 'cuda' and not torch.cuda.is_available():
            kwargs['device'] = model_device
        elif len(args) > 0 and args[0] == 'cuda' and not torch.cuda.is_available():
            args = (model_device,) + args[1:]
        return original_tensor_to(self, *args, **kwargs)
    
    torch.Tensor.to = device_aware_to
    
    try:
        # Run evaluation
        eval_model(args)
        print("\n   ✓ Evaluation complete!")
    except Exception as e:
        print(f"\n   ✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original functions
        model_vqa_module.load_pretrained_model = original_load
        torch.Tensor.cuda = original_tensor_cuda
        torch.Tensor.to = original_tensor_to
    
    # Run POPE evaluation script
    print("\n3. Running POPE metrics calculation...")
    # Use original question file path for annotation dir (not temp file)
    if original_question_file:
        question_file_for_annotation = original_question_file
    else:
        question_file_for_annotation = args.question_file
    
    annotation_dir = os.path.join(os.path.dirname(question_file_for_annotation), 'coco')
    annotation_dir = os.path.abspath(annotation_dir)
    
    if os.path.exists(annotation_dir):
        try:
            # Import from SparseVLMs explicitly to avoid path conflicts
            import importlib.util
            eval_pope_path = os.path.join(sparsevlm_dir, 'llava', 'eval', 'eval_pope.py')
            spec = importlib.util.spec_from_file_location("eval_pope", eval_pope_path)
            eval_pope_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eval_pope_module)
            eval_pope = eval_pope_module.eval_pope
            # Load questions and answers
            questions = [json.loads(line) for line in open(question_file_for_annotation)]
            questions = {question['question_id']: question for question in questions}
            answers = [json.loads(q) for q in open(args.answers_file)]
            
            # Process each category (adversarial, popular, random)
            for file in os.listdir(annotation_dir):
                if file.startswith('coco_pope_') and file.endswith('.json'):
                    category = file[10:-5]  # Extract category name
                    cur_answers = [x for x in answers if questions.get(x['question_id'], {}).get('category') == category]
                    if cur_answers:
                        print(f'   Category: {category}, # samples: {len(cur_answers)}')
                        label_file = os.path.join(annotation_dir, file)
                        # Use baseline F1 from LLaVA-1.5-7B (85.9 according to model zoo)
                        baseline_f1 = 85.9  # Vanilla LLaVA-1.5-7B POPE F1 score
                        eval_pope(cur_answers, label_file, questions_dict=questions, baseline_f1=baseline_f1)
                        print("   " + "=" * 50)
            print("   ✓ POPE metrics calculated")
        except Exception as e:
            print(f"   ⚠ Error calculating POPE metrics: {e}")
            import traceback
            traceback.print_exc()
            print("   Results file saved, but metrics calculation failed")
    else:
        print(f"   ⚠ Annotation directory not found: {annotation_dir}")
        print("   Results file saved, but metrics calculation skipped")
    
    # Clean up temporary file if we created one
    if temp_file:
        try:
            os.unlink(temp_file.name)
        except:
            pass
    
    # Restore original working directory
    os.chdir(original_cwd)
    
    print("\n" + "=" * 60)
    print("✓ Complete!")
    print(f"Results saved to: {args.answers_file}")
    if args.max_samples:
        print(f"⚠ Note: This was a quick test with only {args.max_samples} samples")
    print("=" * 60)

if __name__ == "__main__":
    main()

