"""
Compare VisionZip and SparseVLM results on POPE dataset.
"""
import json
import os
import sys

def load_results(filepath):
    """Load results from JSONL file."""
    results = []
    if not os.path.exists(filepath):
        print(f"⚠ File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def calculate_pope_metrics_silent(answers, label_file, questions_dict=None):
    """Calculate POPE metrics silently (without printing) - same logic as eval_pope."""
    # Load labels - create mapping by question_id, and also by (image, text) for fallback
    labels_by_id = {}
    labels_by_key = {}
    for line in open(label_file, 'r'):
        label_data = json.loads(line)
        question_id = label_data['question_id']
        label = label_data['label']
        image = label_data.get('image', '')
        text = label_data.get('text', '')
        label_val = 0 if label == 'no' else 1
        labels_by_id[question_id] = label_val
        text_key = text.strip().lower().replace('\n', ' ').replace('  ', ' ')
        labels_by_key[(image, text_key)] = label_val
    
    # Process answers
    for answer in answers:
        text = answer['text']
        if text.find('.') != -1:
            text = text.split('.')[0]
        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'
    
    pred_list = []
    label_list = []
    
    for answer in answers:
        # Get prediction
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)
        
        # Get corresponding label by question_id or by image+text
        answer_qid = answer.get('question_id')
        label_val = None
        
        # Method 1: Try direct match by question_id
        if answer_qid is not None and answer_qid in labels_by_id:
            label_val = labels_by_id[answer_qid]
        # Method 2: Try to match by image + prompt text
        elif questions_dict is not None and answer_qid is not None:
            question = questions_dict.get(answer_qid)
            if question is not None:
                image = question.get('image', '')
                prompt = answer.get('prompt', '')
                if prompt:
                    prompt_clean = prompt.split('\n')[0].strip()
                    prompt_key = prompt_clean.strip().lower().replace('\n', ' ').replace('  ', ' ')
                    key = (image, prompt_key)
                    if key in labels_by_key:
                        label_val = labels_by_key[key]
        
        if label_val is not None:
            label_list.append(label_val)
        else:
            pred_list.pop()  # Remove the prediction we just added
    
    # Calculate confusion matrix
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == 1 and label == 1:
            TP += 1
        elif pred == 1 and label == 0:
            FP += 1
        elif pred == 0 and label == 0:
            TN += 1
        elif pred == 0 and label == 1:
            FN += 1
    
    # Calculate metrics
    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0.0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_pope_results(results_file, annotation_dir, question_file):
    """Run POPE evaluation and return metrics."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'LLaVA')))
    
    if not os.path.exists(results_file):
        print(f"⚠ Results file not found: {results_file}")
        return None
    
    # Load questions as dict for proper matching
    questions = {}
    with open(question_file, 'r') as f:
        for line in f:
            q = json.loads(line)
            questions[q['question_id']] = q
    
    # Load answers
    answers = []
    with open(results_file, 'r') as f:
        for line in f:
            answers.append(json.loads(line))
    
    # Process each category (adversarial, popular, random)
    all_metrics = {}
    for file in os.listdir(annotation_dir):
        if not file.startswith('coco_pope_') or not file.endswith('.json'):
            continue
        
        category = file[10:-5]  # Extract category name
        cur_answers = [x for x in answers if questions.get(x['question_id'], {}).get('category') == category]
        label_file = os.path.join(annotation_dir, file)
        
        if not cur_answers:
            continue
        
        # Calculate metrics silently
        try:
            metrics = calculate_pope_metrics_silent(cur_answers, label_file, questions_dict=questions)
            all_metrics[category] = metrics
        except Exception as e:
            print(f"   ⚠ Error evaluating category {category}: {e}")
    
    # Calculate average across categories
    if all_metrics:
        avg_metrics = {
            'accuracy': sum(m.get('accuracy', 0) for m in all_metrics.values()) / len(all_metrics),
            'precision': sum(m.get('precision', 0) for m in all_metrics.values()) / len(all_metrics),
            'recall': sum(m.get('recall', 0) for m in all_metrics.values()) / len(all_metrics),
            'f1': sum(m.get('f1', 0) for m in all_metrics.values()) / len(all_metrics),
        }
        return avg_metrics
    
    return None

def main():
    print("=" * 60)
    print("VisionZip vs SparseVLM Comparison")
    print("=" * 60)
    
    # Updated paths - use playground/eval/pope (not playground/data/eval/pope)
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'LLaVA', 'playground', 'eval', 'pope', 'answers')
    
    # File paths
    visionzip_file = os.path.join(base_dir, 'visionzip-v1.5-7b.jsonl')
    sparsevlm_file = os.path.join(base_dir, 'sparsevlm-v1.5-7b.jsonl')
    
    print("\n1. Loading VisionZip results...")
    visionzip_results = load_results(visionzip_file)
    if visionzip_results:
        print(f"   ✓ Loaded {len(visionzip_results)} VisionZip results")
    else:
        print(f"   ⚠ VisionZip results not found at: {visionzip_file}")
    
    print("\n2. Loading SparseVLM results...")
    sparsevlm_results = load_results(sparsevlm_file)
    if sparsevlm_results:
        print(f"   ✓ Loaded {len(sparsevlm_results)} SparseVLM results")
    else:
        print(f"   ⚠ SparseVLM results not found at: {sparsevlm_file}")
    
    print("\n3. Evaluating metrics...")
    print("-" * 60)
    
    # Get paths - updated to use playground/eval/pope
    annotation_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'LLaVA', 'playground', 'eval', 'pope', 'coco')
    question_file = os.path.join(os.path.dirname(__file__), '..', 'models', 'LLaVA', 'playground', 'eval', 'pope', 'llava_pope_test.jsonl')
    
    visionzip_metrics = evaluate_pope_results(visionzip_file, annotation_dir, question_file)
    sparsevlm_metrics = evaluate_pope_results(sparsevlm_file, annotation_dir, question_file)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    if visionzip_metrics:
        print("\nVisionZip:")
        print(f"  Accuracy:  {visionzip_metrics.get('accuracy', 'N/A'):.3f}")
        print(f"  Precision: {visionzip_metrics.get('precision', 'N/A'):.3f}")
        print(f"  Recall:    {visionzip_metrics.get('recall', 'N/A'):.3f}")
        print(f"  F1 Score:  {visionzip_metrics.get('f1', 'N/A'):.3f}")
    else:
        print("\nVisionZip: ⚠ Results not available")
    
    if sparsevlm_metrics:
        print("\nSparseVLM:")
        print(f"  Accuracy:  {sparsevlm_metrics.get('accuracy', 'N/A'):.3f}")
        print(f"  Precision: {sparsevlm_metrics.get('precision', 'N/A'):.3f}")
        print(f"  Recall:    {sparsevlm_metrics.get('recall', 'N/A'):.3f}")
        print(f"  F1 Score:  {sparsevlm_metrics.get('f1', 'N/A'):.3f}")
    else:
        print("\nSparseVLM: ⚠ Results not available")
    
    # Comparison table
    if visionzip_metrics and sparsevlm_metrics:
        print("\n" + "=" * 60)
        print("COMPARISON TABLE")
        print("=" * 60)
        print(f"{'Metric':<15} {'VisionZip':<15} {'SparseVLM':<15} {'Difference':<15}")
        print("-" * 60)
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics_to_compare:
            v = visionzip_metrics.get(metric, 0)
            s = sparsevlm_metrics.get(metric, 0)
            diff = v - s
            print(f"{metric.capitalize():<15} {v:<15.3f} {s:<15.3f} {diff:+.3f}")
        
        print("=" * 60)
    
    print("\n✓ Comparison complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

