"""
Evaluate VisionZip on POPE dataset.
This script loads the model, applies VisionZip, and runs evaluation.
"""
import sys
import os
import argparse
import json
import random
import tempfile

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'VisionZip')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'LLaVA')))

import torch
from visionzip import visionzip  # type: ignore
from llava.model.builder import load_pretrained_model  # type: ignore
from llava.mm_utils import get_model_name_from_path  # type: ignore
from llava.eval.model_vqa import eval_model  # type: ignore
import argparse as arg_parse

def main():
    parser = arg_parse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/eval/pope/val2014")
    parser.add_argument("--question-file", type=str, default="./playground/eval/pope/llava_pope_test.jsonl")
    parser.add_argument("--answers-file", type=str, default="./playground/eval/pope/answers/visionzip-v1.5-7b.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--dominant", type=int, default=54)
    parser.add_argument("--contextual", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit to N random samples (for quick testing)")
    
    args = parser.parse_args()
    
    # Change to LLaVA directory first (paths are relative to this)
    llava_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'LLaVA'))
    original_cwd = os.getcwd()
    os.chdir(llava_dir)
    
    # Resolve paths relative to LLaVA directory
    if not os.path.isabs(args.question_file):
        args.question_file = os.path.abspath(args.question_file)
    if not os.path.isabs(args.image_folder):
        args.image_folder = os.path.abspath(args.image_folder)
    if not os.path.isabs(args.answers_file):
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
    
    # Paths are already resolved relative to LLaVA directory (we're already there)
    
    print("=" * 60)
    print("VisionZip POPE Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"VisionZip: dominant={args.dominant}, contextual={args.contextual}")
    print(f"Question file: {args.question_file}")
    print(f"Image folder: {args.image_folder}")
    print(f"Answers file: {args.answers_file}")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    print("   (This may take a few minutes on first run)")
    try:
        model_name = get_model_name_from_path(args.model_path)
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"   Found {num_gpus} GPU(s):")
            for i in range(num_gpus):
                print(f"     GPU {i}: {torch.cuda.get_device_name(i)}")
            # Try 4-bit quantization first (saves memory)
            print("   Attempting to load with 4-bit quantization (lower memory)...")
            try:
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path=args.model_path,
                    model_base=args.model_base,
                    model_name=model_name,
                    load_4bit=True
                )
            except Exception as e:
                print(f"   4-bit failed: {e}")
                print("   Trying GPU without quantization (float16)...")
                print("   Using device_map='auto' to distribute across multiple GPUs...")
                try:
                    tokenizer, model, image_processor, context_len = load_pretrained_model(
                        model_path=args.model_path,
                        model_base=args.model_base,
                        model_name=model_name,
                        device_map="auto"  # Automatically distributes across available GPUs
                    )
                except Exception as e2:
                    print(f"   GPU loading failed: {e2}")
                    print("   Falling back to CPU (will be slow)...")
                    tokenizer, model, image_processor, context_len = load_pretrained_model(
                        model_path=args.model_path,
                        model_base=args.model_base,
                        model_name=model_name,
                        device_map="cpu"
                    )
        else:
            print("   Using CPU (will be slower)")
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=args.model_path,
                model_base=args.model_base,
                model_name=model_name,
                device_map="cpu"
            )
        print("   ✓ Model loaded")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Apply VisionZip
    print(f"\n2. Applying VisionZip (dominant={args.dominant}, contextual={args.contextual})...")
    try:
        model = visionzip(model, dominant=args.dominant, contextual=args.contextual)
        print("   ✓ VisionZip applied")
    except Exception as e:
        print(f"   ✗ Error applying VisionZip: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Monkey-patch the eval_model function to use our model
    print("\n3. Running evaluation...")
    import llava.eval.model_vqa as model_vqa_module  # type: ignore
    
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
    
    # Monkey-patch tensor.cuda() to use model's device
    original_tensor_cuda = torch.Tensor.cuda
    
    def device_aware_cuda(self, device=None):
        """Replace .cuda() to use model's device if CUDA not available"""
        if torch.cuda.is_available() and device is None:
            return original_tensor_cuda(self)
        else:
            return self.to(model_device)
    
    torch.Tensor.cuda = device_aware_cuda
    
    try:
        # Run evaluation
        eval_model(args)
        print("\n   ✓ Evaluation complete!")
    except Exception as e:
        print(f"\n   ✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original function
        model_vqa_module.load_pretrained_model = original_load
        torch.Tensor.cuda = original_tensor_cuda
    
    # Run POPE evaluation script
    print("\n4. Running POPE metrics calculation...")
    # Use original question file path for annotation dir (not temp file)
    if original_question_file:
        question_file_for_annotation = original_question_file
    else:
        question_file_for_annotation = args.question_file
    
    annotation_dir = os.path.join(os.path.dirname(question_file_for_annotation), 'coco')
    annotation_dir = os.path.abspath(annotation_dir)
    
    if os.path.exists(annotation_dir):
        try:
            from llava.eval.eval_pope import eval_pope  # type: ignore
            # Load questions and answers
            questions = [json.loads(line) for line in open(question_file_for_annotation)]
            questions = {question['question_id']: question for question in questions}
            answers = [json.loads(q) for q in open(args.answers_file)]
            
            # Process each category (adversarial, popular, random)
            for file in os.listdir(annotation_dir):
                if file.startswith('coco_pope_') and file.endswith('.json'):
                    category = file[10:-5]  # Extract category name (e.g., 'adversarial' from 'coco_pope_adversarial.json')
                    cur_answers = [x for x in answers if questions.get(x['question_id'], {}).get('category') == category]
                    if cur_answers:
                        print(f'   Category: {category}, # samples: {len(cur_answers)}')
                        label_file = os.path.join(annotation_dir, file)
                        # Use baseline F1 from LLaVA-1.5-7B (85.9 according to model zoo)
                        # For full dataset, calculate average across all categories
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

