import os
import json
import argparse
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment',
                        type=str,
                        required=True)

    args = parser.parse_args()
    return args


def get_gt(data_path):
    GT = {}
    if not os.path.exists(data_path):
        print(f"Warning: GT data path {data_path} does not exist. GT files will be missing.")
        return GT
    
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, 'images')):
            image_path = os.path.join(category_dir, 'images')
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            image_path = qa_path = category_dir
        
        if not os.path.isdir(image_path):
            continue
            
        if not os.path.isdir(qa_path):
            # Try to find questions_answers_YN in parent
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
            if not os.path.exists(qa_path):
                print(f"Warning: No GT directory found for {category}")
                continue
        
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            try:
                for line in open(os.path.join(qa_path, file)):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        question, answer = parts[0], parts[1]
                        GT[(category, file, question)] = answer
            except Exception as e:
                print(f"Warning: Error reading {qa_path}/{file}: {e}")
                continue
    return GT

if __name__ == "__main__":

    args = get_args()

    GT = get_gt(
        data_path='MME_Benchmark_release_version'
    )

    experiment = args.experiment

    result_dir = os.path.join('eval_tool', 'answers', experiment)
    os.makedirs(result_dir, exist_ok=True)

    answers = [json.loads(line) for line in open(os.path.join('answers', f'{experiment}.jsonl'))]

    results = defaultdict(list)
    for answer in answers:
        category = answer['question_id'].split('/')[0]
        file = answer['question_id'].split('/')[-1].split('.')[0] + '.txt'
        question = answer['prompt']
        results[category].append((file, answer['prompt'], answer['text']))

    for category, cate_tups in results.items():
        with open(os.path.join(result_dir, f'{category}.txt'), 'w') as fp:
            for file, prompt, answer in cate_tups:
                original_prompt = prompt
                if 'Answer the question using a single word or phrase.' in prompt:
                    prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
                
                # Try to find GT answer
                gt_ans = None
                prompt_variants = []
                
                if 'Please answer yes or no.' not in prompt:
                    prompt_variants.append(prompt + ' Please answer yes or no.')
                    prompt_variants.append(prompt + '  Please answer yes or no.')
                else:
                    prompt_variants.append(prompt)
                    prompt_variants.append(prompt.replace(' Please answer yes or no.', '  Please answer yes or no.'))
                
                # Try each variant
                for variant in prompt_variants:
                    key = (category, file, variant)
                    if key in GT:
                        gt_ans = GT[key]
                        prompt = variant
                        break
                
                # If still not found, use 'unknown' as GT
                if gt_ans is None:
                    print(f"Warning: GT not found for {category}/{file}: {original_prompt[:50]}...")
                    gt_ans = "unknown"
                    prompt = prompt_variants[0]  # Use first variant
                
                tup = file, prompt, gt_ans, answer
                fp.write('\t'.join(tup) + '\n')
