#!/usr/bin/env python3
"""
Calibration script to find optimal thresholds and weights for ensemble detection.
Tests different parameter combinations and finds the best balance between
recall (catching deepfakes) and precision (avoiding false positives).
"""

import sys
import os
import warnings
import logging

# Suppress NNPACK warnings before importing anything
os.environ['PYTORCH_DISABLE_NNPACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', message='.*NNPACK.*')
warnings.filterwarnings('ignore', message='.*Could not initialize NNPACK.*')
logging.getLogger().setLevel(logging.ERROR)

import json
from pathlib import Path
from typing import Dict, List, Tuple
from deepfake_detector import DeepfakeDetector, EnsembleDetector
import itertools

def evaluate_parameters(
    deepfake_files: List[Path],
    real_files: List[Path],
    threshold: float,
    confidence_threshold: float,
    weights: Dict[str, float],
    verbose: bool = False
) -> Dict:
    """
    Evaluate detection performance with given parameters.
    
    Returns:
        Dictionary with metrics: tp, fp, fn, tn, precision, recall, f1, accuracy
    """
    # Suppress warnings during evaluation
    import contextlib
    
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
    
    @contextlib.contextmanager
    def suppress_stderr():
        """Suppress NNPACK warnings in stderr."""
        old_stderr = sys.stderr
        sys.stderr = FilteredStderr(old_stderr)
        try:
            yield
        finally:
            sys.stderr = old_stderr
    
    # Create detector with custom parameters
    with suppress_stderr():
        ensemble = EnsembleDetector(
            models=['clip_vit', 'face_xray'],
            weights=weights,
            threshold=threshold,
            confidence_threshold=confidence_threshold
        )
        
        from deepfake_detector.ensemble_wrapper import EnsembleImageDetectorWrapper
        image_detector = EnsembleImageDetectorWrapper(ensemble)
        detector = DeepfakeDetector()
        detector.image_detector = image_detector
    
    tp = 0  # True positives (deepfakes correctly detected)
    fp = 0  # False positives (real images flagged as deepfakes)
    fn = 0  # False negatives (deepfakes missed)
    tn = 0  # True negatives (real images correctly identified)
    
    # Test deepfake images
    if verbose:
        print(f"\nTesting {len(deepfake_files)} deepfake images...")
    for file_path in deepfake_files:
        try:
            with suppress_stderr():
                result = detector.detect(str(file_path))
            if result.get('is_deepfake', False):
                tp += 1
            else:
                fn += 1
        except Exception as e:
            if verbose:
                print(f"Error processing {file_path}: {e}")
            fn += 1
    
    # Test real images
    if verbose:
        print(f"Testing {len(real_files)} real images...")
    for file_path in real_files:
        try:
            with suppress_stderr():
                result = detector.detect(str(file_path))
            if result.get('is_deepfake', False):
                fp += 1
            else:
                tn += 1
        except Exception as e:
            if verbose:
                print(f"Error processing {file_path}: {e}")
            tn += 1  # Assume real if error
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'threshold': threshold,
        'confidence_threshold': confidence_threshold,
        'weights': weights,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'deepfake_detected': tp,
        'deepfake_total': len(deepfake_files),
        'real_false_positives': fp,
        'real_total': len(real_files)
    }


def grid_search(
    deepfake_files: List[Path],
    real_files: List[Path],
    threshold_range: List[float],
    confidence_range: List[float],
    weight_combinations: List[Dict[str, float]],
    verbose: bool = False
) -> List[Dict]:
    """
    Perform grid search over parameter space.
    
    Returns:
        List of results sorted by F1 score (best first)
    """
    results = []
    total_combinations = len(threshold_range) * len(confidence_range) * len(weight_combinations)
    current = 0
    
    print(f"Testing {total_combinations} parameter combinations...")
    print(f"Processing {len(deepfake_files)} deepfake + {len(real_files)} real = {len(deepfake_files) + len(real_files)} images per combination")
    estimated_time = (len(deepfake_files) + len(real_files)) * total_combinations * 2 / 60  # ~2 sec per image
    print(f"Estimated time: ~{estimated_time:.0f} minutes")
    print("Progress will be shown below. NNPACK warnings are suppressed.\n")
    sys.stdout.flush()
    
    # Redirect stderr to suppress NNPACK warnings during grid search
    import contextlib
    from io import StringIO
    
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
    
    old_stderr = sys.stderr
    sys.stderr = FilteredStderr(old_stderr)
    
    try:
        for threshold in threshold_range:
            for conf_threshold in confidence_range:
                for weights in weight_combinations:
                    current += 1
                    if current % 5 == 0 or current == 1:
                        print(f"Progress: {current}/{total_combinations} ({current*100//total_combinations}%) - "
                              f"Testing threshold={threshold:.2f}, conf={conf_threshold:.2f}")
                        sys.stdout.flush()
                    
                    result = evaluate_parameters(
                        deepfake_files, real_files,
                        threshold, conf_threshold, weights, verbose=False
                    )
                    results.append(result)
    finally:
        sys.stderr = old_stderr
    
    print(f"\nCompleted testing all {total_combinations} combinations!")
    print("Analyzing results...\n")
    sys.stdout.flush()
    
    # Sort by F1 score (best first)
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    return results


def find_optimal_balance(
    deepfake_files: List[Path],
    real_files: List[Path],
    target_recall: float = 0.80,
    max_false_positive_rate: float = 0.20
) -> Dict:
    """
    Find parameters that achieve target recall while keeping false positives low.
    """
    # Define search space
    threshold_range = [0.45, 0.50, 0.55, 0.60]
    confidence_range = [0.35, 0.40, 0.45, 0.50, 0.55]
    
    # Weight combinations to test
    weight_combinations = [
        # Current weights
        {'clip_vit': 0.50, 'face_xray': 0.25, 'metadata': 0.10, 'camera_pipeline': 0.05},
        # More weight to CLIP-ViT
        {'clip_vit': 0.60, 'face_xray': 0.20, 'metadata': 0.10, 'camera_pipeline': 0.05},
        {'clip_vit': 0.55, 'face_xray': 0.25, 'metadata': 0.10, 'camera_pipeline': 0.05},
        # More weight to Face X-Ray
        {'clip_vit': 0.45, 'face_xray': 0.30, 'metadata': 0.10, 'camera_pipeline': 0.05},
        # Less weight to metadata
        {'clip_vit': 0.55, 'face_xray': 0.30, 'metadata': 0.05, 'camera_pipeline': 0.05},
        {'clip_vit': 0.60, 'face_xray': 0.25, 'metadata': 0.05, 'camera_pipeline': 0.05},
    ]
    
    print("Performing grid search...")
    results = grid_search(
        deepfake_files, real_files,
        threshold_range, confidence_range, weight_combinations,
        verbose=True
    )
    
    # Filter results by constraints
    filtered = [
        r for r in results
        if r['recall'] >= target_recall and
        (r['fp'] / len(real_files)) <= max_false_positive_rate
    ]
    
    if filtered:
        print(f"\nFound {len(filtered)} combinations meeting criteria:")
        print(f"  Target recall: >= {target_recall:.0%}")
        print(f"  Max false positive rate: <= {max_false_positive_rate:.0%}")
        return filtered[0]  # Best F1 score
    else:
        print(f"\nNo combination meets criteria. Showing best overall:")
        return results[0]


def collect_files(paths: List[str], file_type: str = 'image') -> List[Path]:
    """Collect files from multiple paths (directories or wildcards)."""
    import glob
    
    all_files = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    
    if file_type == 'image':
        extensions = image_extensions
    elif file_type == 'video':
        extensions = video_extensions
    else:
        extensions = image_extensions | video_extensions
    
    for path_str in paths:
        # Expand wildcards
        if '*' in path_str or '?' in path_str:
            expanded = glob.glob(path_str)
            paths_to_check = [Path(p) for p in expanded]
        else:
            paths_to_check = [Path(path_str)]
        
        for path in paths_to_check:
            if path.is_file() and path.suffix.lower() in extensions:
                all_files.append(path)
            elif path.is_dir():
                # Recursively find files
                for ext in extensions:
                    all_files.extend(path.rglob(f'*{ext}'))
                    all_files.extend(path.rglob(f'*{ext.upper()}'))
    
    # Remove duplicates
    return list(set(all_files))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calibrate ensemble detection parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find optimal parameters (multiple directories)
  python calibrate_ensemble.py testing-data/deepfake/ testing-data/pics/
  
  # With wildcards
  python calibrate_ensemble.py testing-data/deepfake/* testing-data/pics/*
  
  # Multiple directories for deepfakes and reals
  python calibrate_ensemble.py testing-data/deepfake/ testing-data/videos/ \\
      --real testing-data/pics/ testing-data/real/
  
  # Test specific parameters
  python calibrate_ensemble.py testing-data/deepfake/ testing-data/pics/ \\
      --threshold 0.50 --confidence 0.45
  
  # Save results to JSON
  python calibrate_ensemble.py testing-data/deepfake/ testing-data/pics/ \\
      --output calibration_results.json
        """
    )
    
    parser.add_argument('deepfake_paths', nargs='+', type=str, 
                       help='Directory(ies) or file(s) containing deepfake media')
    parser.add_argument('--real', nargs='+', type=str, required=True,
                       help='Directory(ies) or file(s) containing real media')
    parser.add_argument('--threshold', type=float, help='Test specific threshold')
    parser.add_argument('--confidence', type=float, help='Test specific confidence threshold')
    parser.add_argument('--output', '-o', type=str, help='Save results to JSON file')
    parser.add_argument('--top-n', type=int, default=10, help='Show top N results (default: 10)')
    parser.add_argument('--target-recall', type=float, default=0.80, help='Target recall (default: 0.80)')
    parser.add_argument('--max-fp-rate', type=float, default=0.20, help='Max false positive rate (default: 0.20)')
    parser.add_argument('--include-videos', action='store_true', 
                       help='Include videos in calibration (default: images only)')
    
    args = parser.parse_args()
    
    # Collect files
    print("Collecting deepfake files...")
    deepfake_files = collect_files(args.deepfake_paths, 
                                   'all' if args.include_videos else 'image')
    
    print("Collecting real files...")
    real_files = collect_files(args.real, 
                              'all' if args.include_videos else 'image')
    
    # Filter by type if needed
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    
    if not args.include_videos:
        deepfake_files = [f for f in deepfake_files if f.suffix.lower() in image_extensions]
        real_files = [f for f in real_files if f.suffix.lower() in image_extensions]
    
    deepfake_images = [f for f in deepfake_files if f.suffix.lower() in image_extensions]
    deepfake_videos = [f for f in deepfake_files if f.suffix.lower() in video_extensions]
    real_images = [f for f in real_files if f.suffix.lower() in image_extensions]
    real_videos = [f for f in real_files if f.suffix.lower() in video_extensions]
    
    print(f"\nFound:")
    print(f"  Deepfake images: {len(deepfake_images)}")
    if args.include_videos:
        print(f"  Deepfake videos: {len(deepfake_videos)}")
    print(f"  Real images: {len(real_images)}")
    if args.include_videos:
        print(f"  Real videos: {len(real_videos)}")
    
    if len(deepfake_images) == 0 or len(real_images) == 0:
        print("Error: Need at least one deepfake and one real image")
        sys.exit(1)
    
    # Use images for calibration (videos are slower and less reliable for calibration)
    # But note: videos will use the same thresholds found from images
    deepfake_files = deepfake_images
    real_files = real_images
    
    if args.include_videos:
        print("\nNote: Videos are included but calibration focuses on images for speed.")
        print("      Found thresholds will apply to both images and videos.")
    
    # Test specific parameters or find optimal
    if args.threshold is not None and args.confidence is not None:
        print(f"\nTesting specific parameters:")
        print(f"  Threshold: {args.threshold}")
        print(f"  Confidence: {args.confidence}")
        
        weights = {'clip_vit': 0.50, 'face_xray': 0.25, 'metadata': 0.10, 'camera_pipeline': 0.05}
        result = evaluate_parameters(
            deepfake_files, real_files,
            args.threshold, args.confidence, weights, verbose=True
        )
        results = [result]
    else:
        # Find optimal parameters
        optimal = find_optimal_balance(
            deepfake_files, real_files,
            target_recall=args.target_recall,
            max_false_positive_rate=args.max_fp_rate
        )
        
        # Get top N results for comparison
        threshold_range = [0.45, 0.50, 0.55, 0.60]
        confidence_range = [0.35, 0.40, 0.45, 0.50, 0.55]
        weight_combinations = [
            {'clip_vit': 0.50, 'face_xray': 0.25, 'metadata': 0.10, 'camera_pipeline': 0.05},
            {'clip_vit': 0.60, 'face_xray': 0.20, 'metadata': 0.10, 'camera_pipeline': 0.05},
            {'clip_vit': 0.55, 'face_xray': 0.25, 'metadata': 0.10, 'camera_pipeline': 0.05},
            {'clip_vit': 0.45, 'face_xray': 0.30, 'metadata': 0.10, 'camera_pipeline': 0.05},
            {'clip_vit': 0.55, 'face_xray': 0.30, 'metadata': 0.05, 'camera_pipeline': 0.05},
            {'clip_vit': 0.60, 'face_xray': 0.25, 'metadata': 0.05, 'camera_pipeline': 0.05},
        ]
        
        all_results = grid_search(
            deepfake_files, real_files,
            threshold_range, confidence_range, weight_combinations,
            verbose=False
        )
        results = all_results[:args.top_n]
        optimal = results[0]
        
        # If videos were provided, test them with optimal parameters
        if args.include_videos and (len(deepfake_videos) > 0 or len(real_videos) > 0):
            print(f"\n" + "="*80)
            print("TESTING OPTIMAL PARAMETERS ON VIDEOS")
            print("="*80)
            
            # Test videos with optimal parameters
            from deepfake_detector import DeepfakeDetector
            from deepfake_detector.ensemble_wrapper import EnsembleImageDetectorWrapper
            
            ensemble = EnsembleDetector(
                models=['clip_vit', 'face_xray'],
                weights=optimal['weights'],
                threshold=optimal['threshold'],
                confidence_threshold=optimal['confidence_threshold']
            )
            detector = DeepfakeDetector()
            detector.image_detector = EnsembleImageDetectorWrapper(ensemble)
            
            video_tp = 0
            video_fp = 0
            video_fn = 0
            video_tn = 0
            
            # Suppress warnings during video testing
            import contextlib
            @contextlib.contextmanager
            def suppress_stderr():
                old_stderr = sys.stderr
                try:
                    with open(os.devnull, 'w') as devnull:
                        sys.stderr = devnull
                        yield
                finally:
                    sys.stderr = old_stderr
            
            for video_path in deepfake_videos:
                try:
                    with suppress_stderr():
                        result = detector.detect(str(video_path))
                    if result.get('is_deepfake', False):
                        video_tp += 1
                    else:
                        video_fn += 1
                except:
                    video_fn += 1
            
            for video_path in real_videos:
                try:
                    with suppress_stderr():
                        result = detector.detect(str(video_path))
                    if result.get('is_deepfake', False):
                        video_fp += 1
                    else:
                        video_tn += 1
                except:
                    video_tn += 1
            
            if len(deepfake_videos) > 0:
                video_recall = video_tp / len(deepfake_videos)
                print(f"Deepfake videos: {video_tp}/{len(deepfake_videos)} detected ({video_recall:.1%})")
            if len(real_videos) > 0:
                video_fp_rate = video_fp / len(real_videos)
                print(f"Real videos: {video_fp}/{len(real_videos)} false positives ({video_fp_rate:.1%})")
    
    # Display results
    print("\n" + "="*80)
    print("OPTIMAL PARAMETERS")
    print("="*80)
    print(f"Threshold: {optimal['threshold']:.2f}")
    print(f"Confidence Threshold: {optimal['confidence_threshold']:.2f}")
    print(f"Weights:")
    for key, value in optimal['weights'].items():
        print(f"  {key}: {value:.2f}")
    print(f"\nPerformance:")
    print(f"  Deepfakes detected: {optimal['tp']}/{optimal['deepfake_total']} ({optimal['recall']:.1%})")
    print(f"  Real images false positives: {optimal['fp']}/{optimal['real_total']} ({optimal['fp']/optimal['real_total']:.1%})")
    print(f"  Precision: {optimal['precision']:.1%}")
    print(f"  Recall: {optimal['recall']:.1%}")
    print(f"  F1 Score: {optimal['f1']:.1%}")
    print(f"  Accuracy: {optimal['accuracy']:.1%}")
    
    if len(results) > 1:
        print(f"\n" + "="*80)
        print(f"TOP {args.top_n} PARAMETER COMBINATIONS")
        print("="*80)
        for i, r in enumerate(results[:args.top_n], 1):
            print(f"\n{i}. Threshold={r['threshold']:.2f}, Conf={r['confidence_threshold']:.2f}")
            print(f"   Weights: CLIP-ViT={r['weights']['clip_vit']:.2f}, FaceXRay={r['weights']['face_xray']:.2f}")
            print(f"   Recall: {r['recall']:.1%} ({r['tp']}/{r['deepfake_total']} deepfakes detected)")
            print(f"   False Positives: {r['fp']}/{r['real_total']} ({r['fp']/r['real_total']:.1%})")
            print(f"   F1: {r['f1']:.1%}, Precision: {r['precision']:.1%}, Accuracy: {r['accuracy']:.1%}")
    
    # Save to JSON if requested
    if args.output:
        output_data = {
            'optimal': optimal,
            'top_results': results[:args.top_n],
            'test_set': {
                'deepfake_count': len(deepfake_files),
                'real_count': len(real_files)
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

