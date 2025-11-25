#!/usr/bin/env python3
"""
Threshold Calibration Tool
Helps find the optimal threshold for your dataset to balance false positives and false negatives.
"""

import json
import argparse
from pathlib import Path
from deepfake_detector import DeepfakeDetector
from collections import defaultdict


def evaluate_threshold(results_file: str, threshold: float, confidence_threshold: float = 0.0):
    """Evaluate performance at a given threshold."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    # Group by expected category
    stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'total': 0})
    
    for result in results:
        file_path = result.get('file_path', '')
        score = result.get('overall_score', 0.0)
        confidence = result.get('confidence', 0.0)
        is_deepfake = result.get('is_deepfake', False)
        
        # Determine expected label from file path
        expected = 'unknown'
        if 'deepfake' in file_path.lower() or 'df' in file_path.lower():
            expected = 'deepfake'
        elif 'real' in file_path.lower():
            expected = 'real'
        
        if expected == 'unknown':
            continue
        
        stats[expected]['total'] += 1
        
        # Apply threshold
        predicted = (
            score > threshold and 
            confidence > confidence_threshold
        )
        
        if expected == 'deepfake':
            if predicted:
                stats[expected]['tp'] += 1
            else:
                stats[expected]['fn'] += 1
        else:  # real
            if predicted:
                stats[expected]['fp'] += 1
            else:
                stats[expected]['tn'] += 1
    
    # Calculate metrics
    metrics = {}
    for category, s in stats.items():
        if s['total'] == 0:
            continue
        
        tp, fp, tn, fn = s['tp'], s['fp'], s['tn'], s['fn']
        total = s['total']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[category] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positives': fp,
            'false_negatives': fn,
            'total': total
        }
    
    return metrics, stats


def find_optimal_threshold(results_file: str):
    """Find optimal threshold by testing different values."""
    print("Finding optimal threshold...")
    print("=" * 60)
    
    best_threshold = 0.5
    best_f1 = 0
    best_metrics = None
    
    # Test thresholds from 0.3 to 0.8
    for threshold in [x / 100.0 for x in range(30, 81, 5)]:
        for conf_threshold in [0.0, 0.3, 0.5, 0.6]:
            metrics, stats = evaluate_threshold(results_file, threshold, conf_threshold)
            
            # Calculate overall F1 (weighted average)
            if 'deepfake' in metrics and 'real' in metrics:
                df_f1 = metrics['deepfake']['f1']
                real_f1 = metrics['real']['f1']
                df_total = metrics['deepfake']['total']
                real_total = metrics['real']['total']
                total = df_total + real_total
                
                if total > 0:
                    weighted_f1 = (df_f1 * df_total + real_f1 * real_total) / total
                    
                    # Also consider false positive rate (we want to minimize this)
                    fp_rate = metrics['real']['false_positives'] / metrics['real']['total'] if metrics['real']['total'] > 0 else 1.0
                    
                    # Combined score: F1 weighted by (1 - false_positive_rate)
                    # This prioritizes reducing false positives
                    combined_score = weighted_f1 * (1 - fp_rate * 0.5)
                    
                    if combined_score > best_f1:
                        best_f1 = combined_score
                        best_threshold = threshold
                        best_conf_threshold = conf_threshold
                        best_metrics = metrics
    
    print(f"\nOptimal Threshold: {best_threshold:.2f}")
    print(f"Optimal Confidence Threshold: {best_conf_threshold:.2f}")
    print(f"Best Combined Score: {best_f1:.3f}")
    print("\n" + "=" * 60)
    print("Performance at Optimal Threshold:")
    print("=" * 60)
    
    for category, m in best_metrics.items():
        print(f"\n{category.upper()}:")
        print(f"  Accuracy: {m['accuracy']:.2%}")
        print(f"  Precision: {m['precision']:.2%}")
        print(f"  Recall: {m['recall']:.2%}")
        print(f"  F1 Score: {m['f1']:.3f}")
        print(f"  False Positives: {m['false_positives']}/{m['total']}")
        print(f"  False Negatives: {m['false_negatives']}/{m['total']}")
    
    return best_threshold, best_conf_threshold


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate detection threshold based on your test results'
    )
    
    parser.add_argument(
        'results_file',
        help='JSON file with detection results'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        help='Test specific threshold value'
    )
    
    parser.add_argument(
        '--find-optimal',
        action='store_true',
        help='Find optimal threshold automatically'
    )
    
    args = parser.parse_args()
    
    if args.find_optimal:
        find_optimal_threshold(args.results_file)
    elif args.threshold is not None:
        metrics, stats = evaluate_threshold(args.results_file, args.threshold)
        print(f"\nPerformance at threshold {args.threshold:.2f}:")
        print("=" * 60)
        for category, m in metrics.items():
            print(f"\n{category.upper()}:")
            print(f"  Accuracy: {m['accuracy']:.2%}")
            print(f"  Precision: {m['precision']:.2%}")
            print(f"  Recall: {m['recall']:.2%}")
            print(f"  F1 Score: {m['f1']:.3f}")
            print(f"  False Positives: {m['false_positives']}/{m['total']}")
            print(f"  False Negatives: {m['false_negatives']}/{m['total']}")
    else:
        find_optimal_threshold(args.results_file)


if __name__ == '__main__':
    main()

