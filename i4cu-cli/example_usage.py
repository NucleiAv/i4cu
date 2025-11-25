"""
Example usage of the Deepfake Detector tool.
"""

from deepfake_detector import DeepfakeDetector
from pathlib import Path
import json

def main():
    # Initialize the detector
    print("Initializing Deepfake Detector...")
    detector = DeepfakeDetector()
    
    # Example 1: Detect a single image
    print("\n" + "="*60)
    print("Example 1: Detecting a single image")
    print("="*60)
    
    # Try to find a test image
    test_image = None
    test_dirs = ['testing-data/real', 'testing-data/deepfake', 'testing-data/pics']
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            images = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
            if images:
                test_image = images[0]
                break
    
    if test_image:
        result = detector.detect(test_image)
        print(f"\nFile: {test_image}")
        print(f"Type: {result.get('file_type', 'unknown')}")
        print(f"Is Deepfake: {result.get('is_deepfake', False)}")
        print(f"Score: {result.get('overall_score', 0):.2%}")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        
        if result.get('warnings'):
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
    else:
        print("No test images found. Please provide a path to an image file.")
    
    # Example 2: Detect a video
    print("\n" + "="*60)
    print("Example 2: Detecting a video")
    print("="*60)
    
    test_video = None
    test_video_paths = ['testing-data', '.']
    
    for test_path in test_video_paths:
        video_path = Path(test_path)
        if video_path.exists():
            videos = list(video_path.glob('*.mp4'))
            if videos:
                test_video = videos[0]
                break
    
    if test_video:
        print(f"\nAnalyzing video: {test_video}")
        print("This may take a while...")
        result = detector.detect(test_video)
        print(f"\nFile: {test_video}")
        print(f"Type: {result.get('file_type', 'unknown')}")
        print(f"Is Deepfake: {result.get('is_deepfake', False)}")
        print(f"Score: {result.get('overall_score', 0):.2%}")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        
        if 'statistics' in result:
            stats = result['statistics']
            print(f"\nVideo Statistics:")
            print(f"  Frames analyzed: {stats.get('frames_analyzed', 0)}")
            print(f"  Deepfake frames: {stats.get('deepfake_frames', 0)}")
    else:
        print("No test videos found.")
    
    # Example 3: Batch processing
    print("\n" + "="*60)
    print("Example 3: Batch processing")
    print("="*60)
    
    test_dir = Path('testing-data/real')
    if test_dir.exists():
        images = list(test_dir.glob('*.jpg'))[:5]  # Limit to 5 for demo
        if images:
            print(f"\nProcessing {len(images)} images...")
            results = detector.detect_batch(images, show_progress=True)
            
            deepfakes = sum(1 for r in results if r.get('is_deepfake', False))
            print(f"\nResults: {deepfakes} deepfake(s) detected out of {len(results)} file(s)")
    
    # Example 4: Save results to JSON
    print("\n" + "="*60)
    print("Example 4: Saving results to JSON")
    print("="*60)
    
    if test_image:
        result = detector.detect(test_image)
        output_file = 'example_result.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {output_file}")

if __name__ == '__main__':
    main()

