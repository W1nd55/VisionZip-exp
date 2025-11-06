import os
import json
import argparse

def eval_pope(answers, label_file, questions_dict=None, baseline_f1=None):
    """
    Evaluate POPE metrics.
    
    Args:
        answers: List of answer dictionaries
        label_file: Path to label file
        questions_dict: Optional dictionary mapping question_id to question data
        baseline_f1: Optional baseline F1 score (e.g., 85.9 for Vanilla LLaVA-1.5-7B).
                    If provided, calculates relative performance percentage.
    """
    # Load labels - create mapping by question_id, and also by (image, text) for fallback
    # The label file has sequential question_ids (1, 2, 3...) that match the question file order
    labels_by_id = {}
    labels_by_key = {}  # key: (image, text) -> (label, question_text)
    for idx, line in enumerate(open(label_file, 'r')):
        label_data = json.loads(line)
        question_id = label_data['question_id']
        label = label_data['label']
        image = label_data.get('image', '')
        text = label_data.get('text', '')
        # Convert label to 0/1
        label_val = 0 if label == 'no' else 1
        labels_by_id[question_id] = (label_val, text)  # Store both label and question text
        # Create key from image and text (normalize text for matching)
        text_key = text.strip().lower().replace('\n', ' ').replace('  ', ' ')
        labels_by_key[(image, text_key)] = (label_val, text)  # Store both label and question text

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
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
    debug_info = []  # Store question, answer, ground truth for verification
    
    for answer in answers:
        # Get prediction
        if answer['text'] == 'no':
            pred_list.append(0)
            pred_text = 'no'
        else:
            pred_list.append(1)
            pred_text = 'yes'
        
        # Get corresponding label by question_id or by image+text
        answer_qid = answer.get('question_id')
        label_val = None
        label_question_text = None  # Question text from label file
        answer_question_text = answer.get('prompt', 'N/A')
        # Clean up question text for display
        if '\n' in answer_question_text:
            answer_question_text = answer_question_text.split('\n')[0]
        
        # Method 1: Try direct match by question_id
        if answer_qid is not None and answer_qid in labels_by_id:
            label_val, label_question_text = labels_by_id[answer_qid]
        # Method 2: Try to match by image + prompt text (from questions_dict)
        elif questions_dict is not None and answer_qid is not None:
            question = questions_dict.get(answer_qid)
            if question is not None:
                image = question.get('image', '')
                prompt = answer.get('prompt', '')
                if prompt:
                    # Remove the instruction suffix "Answer the question using a single word or phrase."
                    # This appears in answers but not in label file
                    prompt_clean = prompt.split('\n')[0].strip()
                    # Normalize prompt text for matching
                    prompt_key = prompt_clean.strip().lower().replace('\n', ' ').replace('  ', ' ')
                    key = (image, prompt_key)
                    if key in labels_by_key:
                        label_val, label_question_text = labels_by_key[key]
        
        if label_val is not None:
            label_list.append(label_val)
            # Store debug info
            gt_text = 'yes' if label_val == 1 else 'no'
            debug_info.append({
                'answer_question': answer_question_text,
                'label_question': label_question_text or 'N/A',
                'prediction': pred_text,
                'ground_truth': gt_text,
                'match': '✓' if pred_text == gt_text else '✗',
                'question_match': '✓' if label_question_text and answer_question_text.lower().strip() == label_question_text.lower().strip() else '✗'
            })
        else:
            # If question_id not found, skip this answer
            print(f"Warning: question_id {answer_qid} not found in labels (skipping)")
            pred_list.pop()  # Remove the prediction we just added
    
    # Print debug info before metrics
    print("\n" + "="*80)
    print("Question-Answer-Ground Truth Verification:")
    print("="*80)
    for i, info in enumerate(debug_info, 1):
        print(f"{i}. Question (from answer): {info['answer_question']}")
        print(f"   Question (from label): {info['label_question']} {info['question_match']}")
        print(f"   Prediction: {info['prediction']} | Ground Truth: {info['ground_truth']} {info['match']}")
        print()
    print("="*80 + "\n")

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list) if len(pred_list) > 0 else 0.0

    TP, TN, FP, FN = 0, 0, 0, 0
    confusion_matrix_details = {'TP': [], 'FP': [], 'TN': [], 'FN': []}
    
    for idx, (pred, label) in enumerate(zip(pred_list, label_list)):
        if pred == pos and label == pos:
            TP += 1
            confusion_matrix_details['TP'].append(idx)
        elif pred == pos and label == neg:
            FP += 1
            confusion_matrix_details['FP'].append(idx)
        elif pred == neg and label == neg:
            TN += 1
            confusion_matrix_details['TN'].append(idx)
        elif pred == neg and label == pos:
            FN += 1
            confusion_matrix_details['FN'].append(idx)

    print(f'\nTotal samples evaluated: {len(pred_list)}')
    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))
    print(f'\nConfusion Matrix Breakdown:')
    print(f'  TP (True Positive): {TP} samples - Predicted "yes", Ground Truth "yes"')
    if confusion_matrix_details['TP']:
        print(f'    Sample indices: {confusion_matrix_details["TP"]}')
    print(f'  FP (False Positive): {FP} samples - Predicted "yes", Ground Truth "no"')
    if confusion_matrix_details['FP']:
        print(f'    Sample indices: {confusion_matrix_details["FP"]}')
    print(f'  TN (True Negative): {TN} samples - Predicted "no", Ground Truth "no"')
    if confusion_matrix_details['TN']:
        print(f'    Sample indices: {confusion_matrix_details["TN"]}')
    print(f'  FN (False Negative): {FN} samples - Predicted "no", Ground Truth "yes"')
    if confusion_matrix_details['FN']:
        print(f'    Sample indices: {confusion_matrix_details["FN"]}')

    # Handle division by zero cases
    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0.0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    
    print(f'\nMetrics Calculation:')
    print(f'  Accuracy = (TP + TN) / Total = ({TP} + {TN}) / {TP + TN + FP + FN} = {acc:.3f} ({acc * 100:.1f}%)')
    if (TP + FP) > 0:
        print(f'  Precision = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision:.3f} ({precision * 100:.1f}%)')
    else:
        print(f'  Precision = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision:.3f} ({precision * 100:.1f}%) [no positive predictions]')
    if (TP + FN) > 0:
        print(f'  Recall = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {recall:.3f} ({recall * 100:.1f}%)')
    else:
        print(f'  Recall = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {recall:.3f} ({recall * 100:.1f}%) [no positive ground truth]')
    if (precision + recall) > 0:
        print(f'  F1 = 2 * Precision * Recall / (Precision + Recall) = 2 * {precision:.3f} * {recall:.3f} / ({precision:.3f} + {recall:.3f}) = {f1:.3f} ({f1 * 100:.1f}%)')
    else:
        print(f'  F1 = 2 * Precision * Recall / (Precision + Recall) = {f1:.3f} ({f1 * 100:.1f}%) [precision or recall is 0]')
        if TP == 0 and FP == 0 and FN == 0:
            print(f'  ⚠ Note: F1=0 because there are no positive cases (no "yes" predictions or ground truth).')
            print(f'    F1 score measures performance on the positive class. When all cases are negative (TN),')
            print(f'    F1 is undefined and set to 0. Accuracy ({acc*100:.1f}%) is more meaningful here.')
    print(f'  Yes ratio = {yes_ratio:.3f} ({yes_ratio * 100:.1f}%) - Proportion of "yes" predictions')
    print()
    print('Accuracy: {:.3f} ({:.1f}%)'.format(acc, acc * 100))
    print('Precision: {:.3f} ({:.1f}%)'.format(precision, precision * 100))
    print('Recall: {:.3f} ({:.1f}%)'.format(recall, recall * 100))
    print('F1 score: {:.3f} ({:.1f}%)'.format(f1, f1 * 100))
    print('Yes ratio: {:.3f} ({:.1f}%)'.format(yes_ratio, yes_ratio * 100))
    
    # Calculate relative performance vs baseline (for paper table format)
    if baseline_f1 is not None and baseline_f1 > 0:
        relative_percentage = (f1 / baseline_f1) * 100.0
        print(f'\nRelative Performance (Paper Table Format):')
        print(f'  Baseline F1 (Vanilla): {baseline_f1:.3f}')
        print(f'  Your F1 Score: {f1:.3f}')
        print(f'  Relative Performance: {relative_percentage:.1f}% (Your F1 / Baseline F1 × 100%)')
        print(f'  This is the percentage shown in VisionZip paper table')
    
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
        print("====================================")
