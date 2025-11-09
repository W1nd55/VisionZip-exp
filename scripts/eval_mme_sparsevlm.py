#!/usr/bin/env python3
"""
MME Evaluation Script for SparseVLM
Evaluates SparseVLM model on MME benchmark (2374 questions across 14 categories)
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
from PIL import Image
import gc

# Add SparseVLM to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'SparseVLMs'))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init


def load_gt_answers(data_dir):
    """Load GT answers from questions_answers_YN directories."""
    gt_dict = {}
    gt_base = os.path.join(data_dir, 'MME_Benchmark_release_version')
    
    if not os.path.exists(gt_base):
        return gt_dict
    
    for category in os.listdir(gt_base):
        category_dir = os.path.join(gt_base, category)
        if not os.path.isdir(category_dir):
            continue
        
        qa_dir = os.path.join(category_dir, 'questions_answers_YN')
        if not os.path.exists(qa_dir):
            continue
        
        for filename in os.listdir(qa_dir):
            if not filename.endswith('.txt'):
                continue
            
            filepath = os.path.join(qa_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            question = parts[0].strip()
                            answer = parts[1].strip()
                            # Store by (category, image_file, question)
                            image_file = filename.replace('.txt', '')
                            # Handle different image extensions
                            for ext in ['.png', '.jpg', '.jpeg']:
                                if image_file.endswith(ext):
                                    image_file = image_file[:-len(ext)]
                                    break
                            gt_dict[(category, image_file, question)] = answer
            except Exception as e:
                continue
    
    return gt_dict

def normalize_text(text):
    """Normalize text for matching (like POPE does)."""
    return text.strip().lower().replace('\n', ' ').replace('  ', ' ').replace('\t', ' ')

def get_gt_for_question(gt_dict, question_id, prompt, category):
    """Get GT answer for a question with robust matching (like POPE)."""
    # Extract image file from question_id
    image_file = question_id.split('/')[-1]
    image_base = image_file.rsplit('.', 1)[0] if '.' in image_file else image_file
    
    # Clean prompt - remove instruction suffix
    clean_prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
    if '\n' in clean_prompt:
        clean_prompt = clean_prompt.split('\n')[0].strip()
    
    # Try multiple variants (like POPE does)
    variants = []
    
    # Variant 1: With "Please answer yes or no."
    if 'Please answer yes or no.' not in clean_prompt:
        variants.append(clean_prompt + ' Please answer yes or no.')
        variants.append(clean_prompt + '  Please answer yes or no.')  # Double space
    else:
        variants.append(clean_prompt)
        variants.append(clean_prompt.replace(' Please answer yes or no.', '  Please answer yes or no.'))
    
    # Variant 2: Without "Please answer yes or no."
    base_prompt = clean_prompt.replace(' Please answer yes or no.', '').replace('  Please answer yes or no.', '').strip()
    variants.append(base_prompt)
    variants.append(base_prompt + ' Please answer yes or no.')
    
    # Try exact matches first
    for variant in variants:
        key = (category, image_base, variant)
        if key in gt_dict:
            return gt_dict[key], variant
    
    # Try normalized matching (like POPE fallback)
    normalized_prompt = normalize_text(clean_prompt)
    for (cat, img, q), answer in gt_dict.items():
        if cat == category and img == image_base:
            normalized_q = normalize_text(q)
            if normalized_prompt == normalized_q:
                return answer, q
    
    return None, None

def eval_mme_sparsevlm(args):
    """Run MME evaluation with SparseVLM progressive sparsification"""
    
    print("=" * 80)
    print("MME Evaluation - SparseVLM")
    print("=" * 80)
    
    disable_torch_init()
    
    # Load GT answers
    print("\n0. Loading GT answers...")
    gt_dict = load_gt_answers(args.data_dir)
    print(f"✓ Loaded GT for {len(gt_dict)} question-answer pairs")
    
    # Load questions
    questions_file = os.path.join(args.data_dir, 'llava_mme.jsonl')
    print(f"\n1. Loading questions from: {questions_file}")
    
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    
    with open(questions_file, 'r') as f:
        all_questions = [json.loads(line) for line in f]
    
    print(f"✓ Loaded {len(all_questions)} questions")
    
    # Limit for testing - sample across categories
    if args.num_samples > 0:
        # Group by category
        from collections import defaultdict
        questions_by_category = defaultdict(list)
        for q in all_questions:
            questions_by_category[q['category']].append(q)
        
        # Sample evenly across categories
        samples_per_category = max(1, args.num_samples // len(questions_by_category))
        questions = []
        for category, cat_questions in questions_by_category.items():
            questions.extend(cat_questions[:samples_per_category])
        
        # If we need more samples, add from remaining
        if len(questions) < args.num_samples:
            remaining = args.num_samples - len(questions)
            for category, cat_questions in questions_by_category.items():
                if len(questions) >= args.num_samples:
                    break
                already_taken = samples_per_category
                questions.extend(cat_questions[already_taken:already_taken + 1])
                remaining -= 1
        
        questions = questions[:args.num_samples]
        categories_in_sample = set(q['category'] for q in questions)
        print(f"⚠ Testing mode: Using {args.num_samples} samples across {len(categories_in_sample)} categories")
        print(f"   Categories: {', '.join(sorted(categories_in_sample))}")
    else:
        questions = all_questions
    
    # Load model ONCE
    print("\n2. Loading SparseVLM model...")
    print(f"   Model: {args.model_path}")
    print(f"   Retained tokens: {args.retained_tokens}")
    print(f"   Sparse layers: {args.sparse_layers}")
    print(f"   Token counts: {args.token_counts}")
    
    model_name = get_model_name_from_path(args.model_path)
    
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base,
            model_name=model_name,
            load_4bit=True,
            device_map="auto"
        )
        print("✓ Model loaded with 4-bit quantization")
    except Exception as e:
        print(f"   4-bit loading failed: {e}")
        print("   Falling back to float16...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base,
            model_name=model_name,
            device_map="auto"
        )
        print("✓ Model loaded with float16")
    
    print("✓ SparseVLM model loaded")
    
    # Get device - handle CPU offloading
    try:
        model_device = next(model.parameters()).device
        if str(model_device) == 'meta':
            model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except (StopIteration, AttributeError):
        model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"✓ Using device: {model_device}")
    
    # Run evaluation
    print("\n3. Running evaluation...")
    answers = []
    verification_info = []  # Store verification info like POPE
    
    model.eval()
    image_folder = os.path.join(args.data_dir, 'MME_Benchmark_release_version')
    
    # Verify image folder exists
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    print(f"✓ Image folder: {image_folder}")
    
    with torch.no_grad():
        for idx, q in enumerate(tqdm(questions, desc="Evaluating")):
            try:
                image_file = q['image']
                # Image path in JSONL is already relative to MME_Benchmark_release_version
                # e.g., "code_reasoning/0020.png" or "artwork/images/14777.jpg"
                image_path = os.path.join(image_folder, image_file)
                question_text = q['text']
                
                if not os.path.exists(image_path):
                    print(f"\n⚠ Skipping {q['question_id']}: image not found: {image_path}")
                    answers.append({
                        'question_id': q['question_id'],
                        'prompt': question_text,
                        'text': "error",
                        'category': q['category']
                    })
                    continue
                
                # Load and process image
                image = Image.open(image_path).convert('RGB')
                image_tensor = process_images([image], image_processor, model.config)[0]
                image_tensor = image_tensor.unsqueeze(0).half().to(model_device)
                
                # Prepare prompt
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + question_text
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + question_text
                
                conv = conv_templates["llava_v1"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                # Tokenize
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                input_ids = input_ids.unsqueeze(0).to(model_device)
                
                # Generate with retained_tokens parameter - match POPE evaluation
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=[image.size],
                        retained_tokens=args.retained_tokens,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=512,
                        use_cache=True
                    )
                
                # Decode full output (like POPE does)
                full_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                # Extract answer part (remove prompt)
                # Use prompt string directly to avoid decoding input_ids which may have invalid token IDs
                prompt_decoded = prompt.strip()
                
                # Try to remove prompt from output
                if full_output.startswith(prompt_decoded):
                    output_text = full_output[len(prompt_decoded):].strip()
                else:
                    # If prompt not found, slice by token length (safer than decoding input_ids)
                    if output_ids.shape[1] > input_ids.shape[1]:
                        # Decode only the new tokens (generated part)
                        try:
                            output_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
                        except (IndexError, ValueError, RuntimeError) as e:
                            # If decoding fails (invalid token IDs), try to extract from full output
                            # by finding where the answer likely starts
                            if prompt_decoded in full_output:
                                output_text = full_output.split(prompt_decoded, 1)[-1].strip()
                            else:
                                # Last resort: use full output
                                output_text = full_output.strip()
                    else:
                        output_text = full_output.strip()
                
                # Handle missing/empty responses (like POPE)
                if not output_text or output_text == "":
                    print(f"\n⚠ [{idx + 1}/{len(questions)}] {q['question_id']}: Empty response!")
                    output_text = "error"
                
                # Get GT answer with robust matching
                gt_answer, matched_question = get_gt_for_question(gt_dict, q['question_id'], question_text, q['category'])
                
                # Clean question text for display (like POPE)
                display_question = question_text
                if '\n' in display_question:
                    display_question = display_question.split('\n')[0]
                
                # Check question match - normalize both by removing "Please answer yes or no." for comparison
                question_match = "✗"
                if matched_question:
                    # Remove "Please answer yes or no." from both for comparison
                    clean_display = normalize_text(display_question.replace('Please answer yes or no.', '').replace('  Please answer yes or no.', ''))
                    clean_matched = normalize_text(matched_question.replace('Please answer yes or no.', '').replace('  Please answer yes or no.', ''))
                    question_match = "✓" if clean_display == clean_matched else "✗"
                
                # Check answer match
                answer_match = None
                if gt_answer:
                    answer_match = "✓" if output_text.lower().strip() == gt_answer.lower().strip() else "✗"
                
                # Store verification info (like POPE)
                verification_info.append({
                    'question_id': q['question_id'],
                    'answer_question': display_question,
                    'label_question': matched_question or 'N/A',
                    'prediction': output_text,
                    'ground_truth': gt_answer or 'N/A',
                    'question_match': question_match,
                    'answer_match': answer_match or 'N/A',
                    'has_gt': gt_answer is not None
                })
                
                # Print question, prediction, and GT (like POPE)
                print(f"\n[{idx + 1}/{len(questions)}] {q['question_id']}")
                print(f"  Question (from answer): {display_question}")
                if matched_question:
                    print(f"  Question (from GT): {matched_question} {question_match}")
                print(f"  Prediction: {output_text}")
                print(f"  GT: {gt_answer if gt_answer else 'N/A'}")
                if answer_match:
                    print(f"  Match: {answer_match}")
                if not gt_answer:
                    print(f"  ⚠ Warning: GT not found for this question")
                
                # Store answer (only if we have GT, like POPE skips missing labels)
                answers.append({
                    'question_id': q['question_id'],
                    'prompt': question_text,
                    'text': output_text,
                    'category': q['category']
                })
                
                # Memory management every 50 samples
                if (idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                
            except Exception as e:
                print(f"\n✗ Error processing {q['question_id']}: {e}")
                import traceback
                traceback.print_exc()
                answers.append({
                    'question_id': q['question_id'],
                    'prompt': q.get('text', ''),
                    'text': "error",
                    'category': q.get('category', 'unknown')
                })
    
    # Print verification summary (like POPE)
    print("\n" + "="*80)
    print("Question-Answer-Ground Truth Verification Summary:")
    print("="*80)
    
    total_with_gt = sum(1 for v in verification_info if v['has_gt'])
    total_without_gt = len(verification_info) - total_with_gt
    question_matches = sum(1 for v in verification_info if v['question_match'] == '✓')
    answer_matches = sum(1 for v in verification_info if v['has_gt'] and v['answer_match'] == '✓')
    empty_responses = sum(1 for v in verification_info if v['prediction'] == 'error')
    
    print(f"Total questions processed: {len(verification_info)}")
    print(f"  Questions with GT: {total_with_gt}")
    print(f"  Questions without GT: {total_without_gt}")
    print(f"  Question matches: {question_matches}/{total_with_gt}")
    print(f"  Answer matches: {answer_matches}/{total_with_gt}")
    print(f"  Empty responses: {empty_responses}")
    
    if total_without_gt > 0:
        print(f"\n⚠ Warning: {total_without_gt} questions have no GT (will be skipped in metrics)")
    
    if empty_responses > 0:
        print(f"⚠ Warning: {empty_responses} questions had empty responses")
    
    # Show first few examples
    print("\nFirst 5 examples:")
    for i, info in enumerate(verification_info[:5], 1):
        print(f"\n{i}. {info['question_id']}")
        print(f"   Question (from answer): {info['answer_question'][:80]}...")
        if info['label_question'] != 'N/A':
            print(f"   Question (from GT): {info['label_question'][:80]}... {info['question_match']}")
        print(f"   Prediction: {info['prediction']}")
        print(f"   GT: {info['ground_truth']}")
        if info['has_gt']:
            print(f"   Match: {info['answer_match']}")
    
    print("\n" + "="*80 + "\n")
    
    # Save results
    print(f"\n4. Saving results...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        for ans in answers:
            f.write(json.dumps(ans) + '\n')
    
    print(f"✓ Results saved to: {args.output_file}")
    
    # Convert to MME format
    print("\n5. Converting to MME format...")
    convert_script = os.path.join(args.data_dir, 'convert_answer_to_mme.py')
    if not os.path.exists(convert_script):
        print(f"⚠ Conversion script not found: {convert_script}")
        print("   Skipping conversion. You can run it manually later.")
    else:
        convert_cmd = f"cd {args.data_dir} && python convert_answer_to_mme.py --experiment {args.experiment_name}"
        result = os.system(convert_cmd)
        if result != 0:
            print(f"⚠ Conversion script failed with exit code {result}")
    
    # Calculate metrics
    print("\n6. Calculating metrics...")
    eval_tool_dir = os.path.join(args.data_dir, 'eval_tool')
    results_dir = os.path.join(eval_tool_dir, 'answers', args.experiment_name)
    calc_script = os.path.join(eval_tool_dir, 'calculation.py')
    
    if not os.path.exists(calc_script):
        print(f"⚠ Calculation script not found: {calc_script}")
        print("   Skipping metrics calculation. You can run it manually later.")
    elif not os.path.exists(results_dir):
        print(f"⚠ Results directory not found: {results_dir}")
        print("   Conversion may have failed. Check conversion step above.")
    else:
        calc_cmd = f"cd {eval_tool_dir} && python calculation.py --results_dir answers/{args.experiment_name}"
        result = os.system(calc_cmd)
        if result != 0:
            print(f"⚠ Calculation script failed with exit code {result}")
    
    print("\n" + "=" * 80)
    print("✓ MME Evaluation Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_file}")
    print(f"Metrics calculated in: {results_dir}")
    print("\nTo compare with VisionZip, run:")
    print("  python scripts/compare_mme_results.py")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SparseVLM on MME benchmark")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b",
                       help="Path to LLaVA model")
    parser.add_argument("--model-base", type=str, default=None,
                       help="Base model path")
    
    # SparseVLM arguments
    parser.add_argument("--retained-tokens", type=int, default=64,
                       help="Final number of retained tokens")
    parser.add_argument("--sparse-layers", type=str, default="2,6,15",
                       help="Comma-separated list of sparse layers")
    parser.add_argument("--token-counts", type=str, default="66,30,17",
                       help="Comma-separated token counts at each sparse layer")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str,
                       default="/u/m/a/maheenabooba/cs769/models/LLaVA/playground/eval/MME",
                       help="Directory containing MME data")
    parser.add_argument("--num-samples", type=int, default=-1,
                       help="Number of samples to evaluate (-1 for all)")
    
    # Output arguments
    parser.add_argument("--output-file", type=str,
                       default="/u/m/a/maheenabooba/cs769/models/LLaVA/playground/eval/MME/answers/sparsevlm-v1.5-7b.jsonl",
                       help="Output file for answers")
    parser.add_argument("--experiment-name", type=str, default="sparsevlm-v1.5-7b",
                       help="Name for this experiment")
    
    args = parser.parse_args()
    
    # Parse lists
    args.sparse_layers = [int(x) for x in args.sparse_layers.split(',')]
    args.token_counts = [int(x) for x in args.token_counts.split(',')]
    
    eval_mme_sparsevlm(args)

