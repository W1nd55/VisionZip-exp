#!/usr/bin/env python3
"""
Compare VisionZip and SparseVLM results on MME dataset.
Reads results from eval_tool/answers directories and compares metrics.
"""
import os
import sys
import subprocess

def calculate_mme_metrics_from_categories(category_metrics):
    """Calculate overall metrics from per-category metrics."""
    if not category_metrics:
        return None
    
    perception_categories = ['existence', 'count', 'position', 'color', 'posters', 
                            'celebrity', 'scene', 'landmark', 'artwork', 'OCR']
    cognition_categories = ['commonsense_reasoning', 'numerical_calculation', 
                           'text_translation', 'code_reasoning']
    
    perception_correct = 0
    perception_total = 0
    cognition_correct = 0
    cognition_total = 0
    total_correct = 0
    total_questions = 0
    
    for category, metrics in category_metrics.items():
        correct = metrics['correct']
        total = metrics['total']
        
        total_correct += correct
        total_questions += total
        
        if category in perception_categories:
            perception_correct += correct
            perception_total += total
        elif category in cognition_categories:
            cognition_correct += correct
            cognition_total += total
    
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    perception_accuracy = (perception_correct / perception_total * 100) if perception_total > 0 else 0
    cognition_accuracy = (cognition_correct / cognition_total * 100) if cognition_total > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'perception_accuracy': perception_accuracy,
        'cognition_accuracy': cognition_accuracy,
        'total_correct': total_correct,
        'total_questions': total_questions
    }

def load_category_metrics(results_dir):
    """Load per-category metrics from result files."""
    category_metrics = {}
    
    if not os.path.exists(results_dir):
        return category_metrics
    
    perception_categories = ['existence', 'count', 'position', 'color', 'posters', 
                            'celebrity', 'scene', 'landmark', 'artwork', 'OCR']
    cognition_categories = ['commonsense_reasoning', 'numerical_calculation', 
                           'text_translation', 'code_reasoning']
    
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
                    
                    if gt_answer == 'unknown' or gt_answer == '':
                        continue
                    
                    total += 1
                    if gt_answer == pred_answer or pred_answer.startswith(gt_answer):
                        correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        category_metrics[category] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    return category_metrics

def main():
    print("=" * 80)
    print("MME EVALUATION - VisionZip vs SparseVLM Comparison")
    print("=" * 80)
    
    # Base directory
    base_dir = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'models', 
        'LLaVA', 
        'playground', 
        'eval', 
        'MME'
    )
    base_dir = os.path.abspath(base_dir)
    
    # Results directories
    visionzip_results_dir = os.path.join(base_dir, 'eval_tool', 'answers', 'visionzip-v1.5-7b')
    sparsevlm_results_dir = os.path.join(base_dir, 'eval_tool', 'answers', 'sparsevlm-v1.5-7b')
    
    print(f"\nBase directory: {base_dir}")
    print(f"\n1. Loading VisionZip results...")
    print(f"   Results dir: {visionzip_results_dir}")
    
    visionzip_category = load_category_metrics(visionzip_results_dir)
    visionzip_metrics = calculate_mme_metrics_from_categories(visionzip_category)
    
    if visionzip_metrics:
        print(f"   ✓ Loaded VisionZip metrics ({visionzip_metrics['total_questions']} questions)")
    else:
        print(f"   ⚠ VisionZip results not found or incomplete")
    
    print(f"\n2. Loading SparseVLM results...")
    print(f"   Results dir: {sparsevlm_results_dir}")
    
    sparsevlm_category = load_category_metrics(sparsevlm_results_dir)
    sparsevlm_metrics = calculate_mme_metrics_from_categories(sparsevlm_category)
    
    if sparsevlm_metrics:
        print(f"   ✓ Loaded SparseVLM metrics ({sparsevlm_metrics['total_questions']} questions)")
    else:
        print(f"   ⚠ SparseVLM results not found or incomplete")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("OVERALL RESULTS COMPARISON")
    print("=" * 80)
    
    if visionzip_metrics and sparsevlm_metrics:
        print(f"\n{'Metric':<25} {'VisionZip':<20} {'SparseVLM':<20} {'Difference':<15}")
        print("-" * 80)
        
        metrics_to_compare = [
            ('overall_accuracy', 'Overall Accuracy (%)'),
            ('perception_accuracy', 'Perception Accuracy (%)'),
            ('cognition_accuracy', 'Cognition Accuracy (%)')
        ]
        
        for metric_key, metric_name in metrics_to_compare:
            v = visionzip_metrics.get(metric_key, 0)
            s = sparsevlm_metrics.get(metric_key, 0)
            diff = v - s
            print(f"{metric_name:<25} {v:<20.2f} {s:<20.2f} {diff:+.2f}")
        
        # Total questions
        v_total = visionzip_metrics.get('total_questions', 0)
        s_total = sparsevlm_metrics.get('total_questions', 0)
        v_correct = visionzip_metrics.get('total_correct', 0)
        s_correct = sparsevlm_metrics.get('total_correct', 0)
        
        print("-" * 80)
        print(f"{'Total Questions':<25} {v_total:<20} {s_total:<20} {v_total - s_total:+d}")
        print(f"{'Total Correct':<25} {v_correct:<20} {s_correct:<20} {v_correct - s_correct:+d}")
        print("=" * 80)
    else:
        if not visionzip_metrics:
            print("\n⚠ VisionZip metrics not available")
        if not sparsevlm_metrics:
            print("\n⚠ SparseVLM metrics not available")
    
    # Per-category comparison
    if visionzip_category and sparsevlm_category:
        print("\n" + "=" * 80)
        print("PER-CATEGORY COMPARISON")
        print("=" * 80)
        
        # Get all categories
        all_categories = sorted(set(list(visionzip_category.keys()) + list(sparsevlm_category.keys())))
        
        perception_categories = ['existence', 'count', 'position', 'color', 'posters', 
                                'celebrity', 'scene', 'landmark', 'artwork', 'OCR']
        cognition_categories = ['commonsense_reasoning', 'numerical_calculation', 
                               'text_translation', 'code_reasoning']
        
        print("\nPERCEPTION TASKS:")
        print("-" * 80)
        print(f"{'Category':<25} {'VisionZip':<15} {'SparseVLM':<15} {'Difference':<15}")
        print("-" * 80)
        
        for cat in perception_categories:
            if cat in visionzip_category and cat in sparsevlm_category:
                v_acc = visionzip_category[cat]['accuracy']
                s_acc = sparsevlm_category[cat]['accuracy']
                diff = v_acc - s_acc
                print(f"{cat:<25} {v_acc:>6.2f}%{'':<8} {s_acc:>6.2f}%{'':<8} {diff:+.2f}%")
        
        print("\nCOGNITION TASKS:")
        print("-" * 80)
        print(f"{'Category':<25} {'VisionZip':<15} {'SparseVLM':<15} {'Difference':<15}")
        print("-" * 80)
        
        for cat in cognition_categories:
            if cat in visionzip_category and cat in sparsevlm_category:
                v_acc = visionzip_category[cat]['accuracy']
                s_acc = sparsevlm_category[cat]['accuracy']
                diff = v_acc - s_acc
                print(f"{cat:<25} {v_acc:>6.2f}%{'':<8} {s_acc:>6.2f}%{'':<8} {diff:+.2f}%")
        
        print("=" * 80)
    
    print("\n✓ Comparison complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

