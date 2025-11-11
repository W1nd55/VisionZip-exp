#!/usr/bin/env python3
"""
MME Evaluation Calculation Script
Calculates accuracy for each category and overall scores
"""
import os
import argparse

def calculate_metrics(results_dir):
    """Calculate MME metrics from result files"""
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print("=" * 80)
    print(f"MME Evaluation Results: {os.path.basename(results_dir)}")
    print("=" * 80)
    
    perception_categories = ['existence', 'count', 'position', 'color', 'posters', 
                            'celebrity', 'scene', 'landmark', 'artwork', 'OCR']
    cognition_categories = ['commonsense_reasoning', 'numerical_calculation', 
                           'text_translation', 'code_reasoning']
    
    perception_scores = {}
    cognition_scores = {}
    total_correct = 0
    total_questions = 0
    
    for filename in os.listdir(results_dir):
        if not filename.endswith('.txt'):
            continue
            
        category = filename.replace('.txt', '')
        filepath = os.path.join(results_dir, filename)
        
        correct = 0
        total = 0
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    gt_answer = parts[2].strip().lower()
                    pred_answer = parts[3].strip().lower()
                    
                    # Skip entries with unknown GT
                    if gt_answer == 'unknown' or gt_answer == '':
                        continue
                    
                    total += 1
                    if gt_answer == pred_answer or pred_answer.startswith(gt_answer):
                        correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        if category in perception_categories:
            perception_scores[category] = (correct, total, accuracy)
        elif category in cognition_categories:
            cognition_scores[category] = (correct, total, accuracy)
        
        total_correct += correct
        total_questions += total
    
    # Print detailed results
    print("\nPERCEPTION TASKS:")
    print("-" * 80)
    print(f"{'Category':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 80)
    
    perception_correct = 0
    perception_total = 0
    for cat in perception_categories:
        if cat in perception_scores:
            correct, total, acc = perception_scores[cat]
            perception_correct += correct
            perception_total += total
            print(f"{cat:<25} {correct:<10} {total:<10} {acc:>6.2f}%")
    
    perception_acc = (perception_correct / perception_total * 100) if perception_total > 0 else 0
    print("-" * 80)
    print(f"{'PERCEPTION TOTAL':<25} {perception_correct:<10} {perception_total:<10} {perception_acc:>6.2f}%")
    
    print("\nCOGNITION TASKS:")
    print("-" * 80)
    print(f"{'Category':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 80)
    
    cognition_correct = 0
    cognition_total = 0
    for cat in cognition_categories:
        if cat in cognition_scores:
            correct, total, acc = cognition_scores[cat]
            cognition_correct += correct
            cognition_total += total
            print(f"{cat:<25} {correct:<10} {total:<10} {acc:>6.2f}%")
    
    cognition_acc = (cognition_correct / cognition_total * 100) if cognition_total > 0 else 0
    print("-" * 80)
    print(f"{'COGNITION TOTAL':<25} {cognition_correct:<10} {cognition_total:<10} {cognition_acc:>6.2f}%")
    
    print("\nOVERALL SUMMARY:")
    print("=" * 80)
    overall_acc = (total_correct / total_questions * 100) if total_questions > 0 else 0
    print(f"Total Correct: {total_correct}/{total_questions}")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"Perception Accuracy: {perception_acc:.2f}%")
    print(f"Cognition Accuracy: {cognition_acc:.2f}%")
    print("=" * 80)
    
    return {
        'overall_accuracy': overall_acc,
        'perception_accuracy': perception_acc,
        'cognition_accuracy': cognition_acc,
        'total_correct': total_correct,
        'total_questions': total_questions
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing the result txt files')
    args = parser.parse_args()
    
    calculate_metrics(args.results_dir)

