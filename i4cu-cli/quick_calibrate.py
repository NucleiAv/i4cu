#!/usr/bin/env python3
"""
Quick calibration script - faster approach using sampling and smart search.
Finds good parameters in minutes instead of hours.
"""

import sys
import os
import warnings
import logging

# Suppress NNPACK warnings
os.environ['PYTORCH_DISABLE_NNPACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', message='.*NNPACK.*')
warnings.filterwarnings('ignore', message='.*Could not initialize NNPACK.*')
logging.getLogger().setLevel(logging.ERROR)

import json
from pathlib import Path
from typing import Dict, List, Tuple
from deepfake_detector import DeepfakeDetector, EnsembleDetector
from deepfake_detector.ensemble_wrapper import EnsembleImageDetectorWrapper
import random

class FilteredStderr:
    """Filter stderr to suppress NNPACK warnings."""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, message):
        if 'NNPACK' not in message and 'Could not initialize NNPACK' not in message:
            self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

sys.stderr = FilteredStderr(sys.stderr)


def evaluate_quick(
    deepfake_sample: List[Path],
    real_sample: List[Path],
    threshold: float,
    confidence_threshold: float,
    weights: Dict[str, float]
) -> Dict:
    """Quick evaluation on sample."""
    ensemble = EnsembleDetector(
        models=['clip_vit', 'face_xray'],
        weights=weights,
        threshold=threshold,
        confidence_threshold=confidence_threshold
    )
    
    detector = DeepfakeDetector()
    detector.image_detector = EnsembleImageDetectorWrapper(ensemble)
    
    tp = fp = fn = tn = 0
    
    for file_path in deepfake_sample:
        try:
            result = detector.detect(str(file_path))
            if result.get('is_deepfake', False):
                tp += 1
            else:
                fn += 1
        except:
            fn += 1
    
    for file_path in real_sample:
        try:
            result = detector.detect(str(file_path))
            if result.get('is_deepfake', False):
                fp += 1
            else:
                tn += 1
        except:
            tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'threshold': threshold,
        'confidence_threshold': confidence_threshold,
        'weights': weights,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fp_rate': fp / len(real_sample) if real_sample else 0
    }


def smart_search(
    deepfake_files: List[Path],
    real_files: List[Path],
    target_recall: float = 0.80,
    max_fp_rate: float = 0.20
) -> Dict:
    """
    Smart search using sampling and focused parameter ranges.
    Much faster than full grid search.
    """
    # Use sampling for speed (test on subset)
    sample_size = min(30, len(deepfake_files), len(real_files))
    deepfake_sample = random.sample(deepfake_files, sample_size)
    real_sample = random.sample(real_files, sample_size)
    
    print(f"Using {sample_size} samples from each set for quick testing...")
    print("This will take ~5-10 minutes instead of 13+ hours.\n")
    
    # Focused parameter ranges based on current performance
    # Current: 74% recall, 44% false positives
    # Need: 80-90% recall, 10-20% false positives
    
    # Test fewer, more targeted combinations
    threshold_range = [0.48, 0.50, 0.52, 0.54, 0.56]  # Around current 0.50
    confidence_range = [0.40, 0.42, 0.44, 0.46, 0.48]  # Around current 0.45
    
    # Test key weight combinations
    weight_combinations = [
        {'clip_vit': 0.55, 'face_xray': 0.30, 'metadata': 0.10, 'camera_pipeline': 0.05},  # More CLIP-ViT
        {'clip_vit': 0.60, 'face_xray': 0.25, 'metadata': 0.10, 'camera_pipeline': 0.05},  # Even more CLIP-ViT
        {'clip_vit': 0.50, 'face_xray': 0.35, 'metadata': 0.10, 'camera_pipeline': 0.05},  # More Face X-Ray
        {'clip_vit': 0.58, 'face_xray': 0.27, 'metadata': 0.10, 'camera_pipeline': 0.05},  # Balanced
        {'clip_vit': 0.62, 'face_xray': 0.23, 'metadata': 0.10, 'camera_pipeline': 0.05},  # CLIP-ViT heavy
    ]
    
    total = len(threshold_range) * len(confidence_range) * len(weight_combinations)
    print(f"Testing {total} focused combinations...\n")
    
    results = []
    current = 0
    
    for threshold in threshold_range:
        for conf in confidence_range:
            for weights in weight_combinations:
                current += 1
                if current % 5 == 0 or current == 1:
                    print(f"Progress: {current}/{total} ({current*100//total}%)")
                    sys.stdout.flush()
                
                result = evaluate_quick(deepfake_sample, real_sample, threshold, conf, weights)
                results.append(result)
    
    # Sort by F1, but prefer results meeting criteria
    def score_result(r):
        meets_recall = r['recall'] >= target_recall
        meets_fp = r['fp_rate'] <= max_fp_rate
        if meets_recall and meets_fp:
            return r['f1'] + 0.5  # Boost if meets criteria
        return r['f1']
    
    results.sort(key=score_result, reverse=True)
    
    print(f"\nCompleted! Testing best parameters on full dataset...\n")
    
    # Test top 3 on full dataset
    best = None
    best_score = -1
    
    for i, result in enumerate(results[:3]):
        print(f"Testing candidate {i+1}: threshold={result['threshold']:.2f}, "
              f"conf={result['confidence_threshold']:.2f}")
        sys.stdout.flush()
        
        full_result = evaluate_quick(deepfake_files, real_files,
                                     result['threshold'], result['confidence_threshold'],
                                     result['weights'])
        
        score = full_result['f1']
        if full_result['recall'] >= target_recall and full_result['fp_rate'] <= max_fp_rate:
            score += 0.3  # Bonus for meeting criteria
        
        if score > best_score:
            best_score = score
            best = full_result
    
    return best if best else results[0]


def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Quick calibration (5-10 minutes)')
    parser.add_argument('deepfake_paths', nargs='+', help='Deepfake files/directories')
    parser.add_argument('--real', nargs='+', required=True, help='Real files/directories')
    parser.add_argument('--output', '-o', help='Save results to JSON')
    parser.add_argument('--target-recall', type=float, default=0.80, help='Target recall')
    parser.add_argument('--max-fp-rate', type=float, default=0.20, help='Max false positive rate')
    
    args = parser.parse_args()
    
    # Collect files
    def collect(paths):
        files = []
        for p in paths:
            if '*' in p:
                files.extend([Path(f) for f in glob.glob(p)])
            else:
                path = Path(p)
                if path.is_file():
                    files.append(path)
                elif path.is_dir():
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                        files.extend(path.rglob(f'*{ext}'))
        return list(set(files))
    
    deepfake_files = collect(args.deepfake_paths)
    real_files = collect(args.real)
    
    print(f"Found {len(deepfake_files)} deepfake images")
    print(f"Found {len(real_files)} real images\n")
    
    if len(deepfake_files) == 0 or len(real_files) == 0:
        print("Error: Need at least one deepfake and one real image")
        sys.exit(1)
    
    # Run smart search
    optimal = smart_search(deepfake_files, real_files,
                          target_recall=args.target_recall,
                          max_fp_rate=args.max_fp_rate)
    
    # Display results
    print("\n" + "="*80)
    print("OPTIMAL PARAMETERS (Quick Calibration)")
    print("="*80)
    print(f"Threshold: {optimal['threshold']:.2f}")
    print(f"Confidence Threshold: {optimal['confidence_threshold']:.2f}")
    print(f"Weights:")
    for key, value in optimal['weights'].items():
        print(f"  {key}: {value:.2f}")
    print(f"\nPerformance on full dataset:")
    print(f"  Deepfakes detected: {optimal['tp']}/{len(deepfake_files)} ({optimal['recall']:.1%})")
    print(f"  Real images false positives: {optimal['fp']}/{len(real_files)} ({optimal['fp_rate']:.1%})")
    print(f"  Precision: {optimal['precision']:.1%}")
    print(f"  Recall: {optimal['recall']:.1%}")
    print(f"  F1 Score: {optimal['f1']:.1%}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(optimal, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

