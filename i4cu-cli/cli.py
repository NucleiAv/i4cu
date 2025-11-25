#!/usr/bin/env python3
"""
Command-line interface for the Deepfake Detector tool.
"""

import argparse
import json
import sys
import warnings
import logging
import os

# Suppress NNPACK warnings (harmless - PyTorch falls back to default CPU implementation)
# These warnings come from PyTorch's C++ backend, so we need multiple approaches
# Set environment variables BEFORE importing anything PyTorch-related
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_DISABLE_NNPACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'  # Disable OpenMP to avoid NNPACK

warnings.filterwarnings('ignore', message='.*NNPACK.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='.*Could not initialize NNPACK.*')

# Suppress PyTorch NNPACK warnings at logging level
logging.getLogger().setLevel(logging.ERROR)

# Context manager to suppress NNPACK warnings from stderr
import contextlib
import io

class FilteredStderr:
    """Filter stderr to suppress NNPACK warnings."""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = ''
    
    def write(self, message):
        # Immediately filter NNPACK warnings
        if 'NNPACK' in message or 'Could not initialize NNPACK' in message:
            return  # Drop the message entirely
        
        # Buffer messages to catch multi-line warnings
        self.buffer += message
        # Check if buffer contains NNPACK warning
        if 'NNPACK' in self.buffer:
            # Clear buffer if it's just NNPACK warnings
            if '\n' in self.buffer:
                lines = self.buffer.split('\n')
                filtered_lines = [line for line in lines if 'NNPACK' not in line and 'Could not initialize NNPACK' not in line]
                self.buffer = '\n'.join(filtered_lines) if filtered_lines else ''
            else:
                self.buffer = ''
        # Write non-NNPACK content
        if self.buffer and 'NNPACK' not in self.buffer and 'Could not initialize NNPACK' not in self.buffer:
            self.original_stderr.write(self.buffer)
            self.buffer = ''
    
    def flush(self):
        if self.buffer and 'NNPACK' not in self.buffer and 'Could not initialize NNPACK' not in self.buffer:
            self.original_stderr.write(self.buffer)
            self.buffer = ''
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

@contextlib.contextmanager
def suppress_nnpack_warnings():
    """Suppress NNPACK warnings while keeping other stderr output."""
    old_stderr = sys.stderr
    sys.stderr = FilteredStderr(old_stderr)
    try:
        yield
    finally:
        sys.stderr = old_stderr

import numpy as np
from pathlib import Path
from deepfake_detector import DeepfakeDetector


def format_result(result: dict, verbose: bool = False) -> str:
    """Format detection result for display."""
    output = []
    
    if 'error' in result:
        output.append(f"❌ Error: {result['error']}")
        return "\n".join(output)
    
    file_path = result.get('file_path', 'Unknown')
    file_type = result.get('file_type', 'unknown')
    is_deepfake = result.get('is_deepfake', False)
    score = result.get('overall_score', 0.0)
    confidence = result.get('confidence', 0.0)
    
    # Header
    status = "🔴 DEEPFAKE DETECTED" if is_deepfake else "✅ Likely Real"
    output.append(f"\n{'='*60}")
    output.append(f"File: {file_path}")
    output.append(f"Type: {file_type.upper()}")
    output.append(f"Status: {status}")
    output.append(f"Score: {score:.2%} (Confidence: {confidence:.2%})")
    output.append(f"{'='*60}")
    
    if verbose:
        # EXIF Analysis (for images)
        if 'exif_analysis' in result:
            exif = result['exif_analysis']
            output.append("\n📋 EXIF Analysis:")
            if exif.get('exif_data'):
                for key, value in exif['exif_data'].items():
                    output.append(f"  {key}: {value}")
            if exif.get('findings'):
                for finding in exif['findings']:
                    output.append(f"  ⚠️  {finding}")
            output.append(f"  Suspicious Score: {exif.get('suspicious_score', 0):.2%}")
        
        # OCR Analysis (for images)
        if 'ocr_analysis' in result:
            ocr = result['ocr_analysis']
            output.append("\n🔍 OCR Analysis:")
            if ocr.get('text_found'):
                output.append(f"  Text found: {ocr.get('text_content', '')[:100]}...")
            if ocr.get('findings'):
                for finding in ocr['findings']:
                    output.append(f"  ⚠️  {finding}")
            output.append(f"  Suspicious Score: {ocr.get('suspicious_score', 0):.2%}")
        
        # Video Info
        if 'video_info' in result:
            video_info = result['video_info']
            output.append("\n🎬 Video Information:")
            for key, value in video_info.items():
                if key != 'error':
                    output.append(f"  {key}: {value}")
        
        # Audio Info
        if 'audio_info' in result:
            audio_info = result['audio_info']
            output.append("\n🎵 Audio Information:")
            for key, value in audio_info.items():
                if key != 'error':
                    output.append(f"  {key}: {value}")
        
        # Feature Analysis (for audio)
        if 'feature_analysis' in result:
            features = result['feature_analysis']
            output.append("\n🎵 Feature Analysis:")
            if features.get('findings'):
                for finding in features['findings']:
                    output.append(f"  ⚠️  {finding}")
            output.append(f"  Suspicious Score: {features.get('suspicious_score', 0):.2%}")
        
        # ML Prediction
        if 'ml_prediction' in result:
            ml = result['ml_prediction']
            output.append("\n🤖 ML Model Prediction:")
            output.append(f"  Score: {ml.get('score', 0):.2%}")
            output.append(f"  Confidence: {ml.get('confidence', 0):.2%}")
            if 'prediction' in ml:
                output.append(f"  Prediction: {ml['prediction']}")
            
            # Show ensemble model scores if available
            if 'model_scores' in ml:
                output.append("\n  Ensemble Model Scores:")
                for model_name, score in ml['model_scores'].items():
                    output.append(f"    {model_name}: {score:.2%}")
                if 'metadata_score' in ml:
                    output.append(f"    Metadata Integrity: {ml['metadata_score']:.2%}")
                if 'camera_pipeline_score' in ml:
                    output.append(f"    Camera Pipeline: {ml['camera_pipeline_score']:.2%}")
        
        # Statistics (for videos)
        if 'statistics' in result:
            stats = result['statistics']
            output.append("\n📊 Statistics:")
            for key, value in stats.items():
                output.append(f"  {key}: {value}")
        
        # Warnings
        if result.get('warnings'):
            output.append("\n⚠️  Warnings:")
            for warning in result['warnings']:
                output.append(f"  - {warning}")
    
    return "\n".join(output)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Deepfake Detector - Detect deepfakes in images, videos, and audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect a single image
  python cli.py image.jpg
  
  # Detect multiple files
  python cli.py image1.jpg image2.jpg video.mp4
  
  # Detect all files in a directory
  python cli.py --directory testing-data
  
  # Detect multiple directories
  python cli.py dir1/ dir2/ --output results.json
  
  # Save results to JSON
  python cli.py image.jpg --output results.json
  
  # Verbose output
  python cli.py image.jpg --verbose
  
  # JSON-only output
  python cli.py image.jpg --json --output results.json
        """
    )
    
    parser.add_argument(
        'input',
        nargs='*',
        help='Input file(s) or directory(ies) to analyze (can specify multiple)'
    )
    
    parser.add_argument(
        '--directory', '-d',
        action='store_true',
        help='Treat input as a directory and process all supported files'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        default=True,
        help='Recursively search directories (default: True)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file to save results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed analysis results'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON only (no formatted text)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-essential output'
    )

    parser.add_argument(
        '--force-audio',
        action='store_true',
        help='Treat media files (e.g. .mp4) as audio-only and run the audio deepfake detector'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='auto',
        choices=['auto', 'ensemble', 'clip_vit', 'uia_vit', 'face_xray', 'xception', 'mesonet', 'efficientnet', 'heuristic'],
        help='Model type to use (default: auto). Use "ensemble" for best accuracy (combines multiple models)'
    )
    
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    # Initialize detector with specified model (suppress NNPACK warnings during init)
    with suppress_nnpack_warnings():
        detector = DeepfakeDetector(model_type=args.model)
    
    if not args.quiet and args.model != 'auto':
        print(f"Using model type: {args.model}")
    
    # Process inputs (can be multiple files/directories)
    # Expand wildcards if shell didn't do it
    import glob
    expanded_inputs = []
    for inp in args.input:
        if '*' in inp or '?' in inp:
            expanded = glob.glob(inp)
            if expanded:
                expanded_inputs.extend(expanded)
            else:
                expanded_inputs.append(inp)
        else:
            expanded_inputs.append(inp)
    
    results = []
    input_paths = [Path(p) for p in expanded_inputs]
    
    # Show progress for multiple files
    if not args.quiet and len(input_paths) > 1:
        print(f"Found {len(input_paths)} file(s) to process...")
        sys.stdout.flush()
    
    for idx, input_path in enumerate(input_paths, 1):
        if not input_path.exists():
            results.append({
                'error': f"Path does not exist: {input_path}",
                'file_path': str(input_path)
            })
            if not args.quiet:
                print(f"Warning: Path does not exist: {input_path}", file=sys.stderr)
            continue
        
        if args.directory or input_path.is_dir():
            # Process directory
            if not args.quiet:
                print(f"Processing directory: {input_path}")
            with suppress_nnpack_warnings():
                dir_results = detector.detect_directory(input_path, recursive=args.recursive)
            results.extend(dir_results)
        else:
            # Process single file
            if not args.quiet:
                if len(input_paths) > 1:
                    print(f"\n[{idx}/{len(input_paths)}] Processing: {input_path.name}")
                else:
                    print(f"Processing file: {input_path}")
                sys.stdout.flush()

            try:
                with suppress_nnpack_warnings():
                    # If user wants to force audio processing (e.g. audio-only MP4),
                    # bypass normal type detection and call the audio detector directly.
                    if args.force_audio:
                        result = detector.audio_detector.detect(str(input_path))
                    else:
                        result = detector.detect(input_path)
                results.append(result)
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                sys.exit(1)
            except Exception as e:
                if not args.quiet:
                    print(f"Error processing {input_path}: {e}", file=sys.stderr)
                results.append({
                    'error': str(e),
                    'file_path': str(input_path)
                })
    
    # Output results
    if args.json:
        # JSON output only
        output_data = {
            'results': results,
            'summary': {
                'total_files': len(results),
                'deepfakes_detected': sum(1 for r in results if r.get('is_deepfake', False)),
                'errors': sum(1 for r in results if 'error' in r)
            }
        }
        print(json.dumps(output_data, indent=2))
    else:
        # Formatted output
        for result in results:
            if not args.quiet:
                print(format_result(result, verbose=args.verbose))
        
        # Summary
        if len(results) > 1 and not args.quiet:
            deepfakes = sum(1 for r in results if r.get('is_deepfake', False))
            errors = sum(1 for r in results if 'error' in r)
            print(f"\n{'='*60}")
            print(f"Summary: {deepfakes} deepfake(s) detected out of {len(results)} file(s)")
            if errors > 0:
                print(f"Errors: {errors}")
            print(f"{'='*60}")
    
    # Save to file if requested
    if args.output:
        # Convert all values to JSON-serializable types
        def make_json_serializable(obj):
            """Recursively convert numpy types and other non-serializable types to native Python types."""
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, (str, int, float)) or obj is None:
                return obj
            else:
                # Try to convert to string as fallback
                return str(obj)
        
        output_data = {
            'results': make_json_serializable(results),
            'summary': {
                'total_files': len(results),
                'deepfakes_detected': sum(1 for r in results if r.get('is_deepfake', False)),
                'errors': sum(1 for r in results if 'error' in r)
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        if not args.quiet:
            print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

