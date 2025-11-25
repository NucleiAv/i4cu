# Calibration Guide - Finding Optimal Parameters

## Overview

The `calibrate_ensemble.py` script helps you find the optimal thresholds and weights for your specific dataset. It tests different parameter combinations and finds the best balance between:
- **Recall**: Catching deepfakes (true positives)
- **Precision**: Avoiding false positives

## Quick Start

### Basic Usage

```bash
# Find optimal parameters for your dataset
python calibrate_ensemble.py testing-data/deepfake/ testing-data/pics/
```

This will:
1. Test multiple threshold/confidence/weight combinations
2. Find the best balance for your data
3. Show top 10 results

### Test Specific Parameters

```bash
# Test specific threshold and confidence
python calibrate_ensemble.py testing-data/deepfake/ testing-data/pics/ \
    --threshold 0.50 --confidence 0.45
```

### Save Results

```bash
# Save results to JSON file
python calibrate_ensemble.py testing-data/deepfake/ testing-data/pics/ \
    --output calibration_results.json
```

### Custom Targets

```bash
# Target 85% recall with max 15% false positives
python calibrate_ensemble.py testing-data/deepfake/ testing-data/pics/ \
    --target-recall 0.85 --max-fp-rate 0.15
```

## Understanding the Output

### Metrics Explained

- **Recall**: Percentage of deepfakes correctly detected
  - Higher = catches more deepfakes
  - Lower = misses more deepfakes
  
- **Precision**: Percentage of flagged items that are actually deepfakes
  - Higher = fewer false positives
  - Lower = more false positives
  
- **F1 Score**: Harmonic mean of precision and recall
  - Higher = better overall balance
  
- **Accuracy**: Overall correctness
  - (True Positives + True Negatives) / Total

### Example Output

```
OPTIMAL PARAMETERS
================================================================================
Threshold: 0.52
Confidence Threshold: 0.48
Weights:
  clip_vit: 0.55
  face_xray: 0.30
  metadata: 0.10
  camera_pipeline: 0.05

Performance:
  Deepfakes detected: 85/100 (85.0%)
  Real images false positives: 12/101 (11.9%)
  Precision: 87.6%
  Recall: 85.0%
  F1 Score: 86.3%
  Accuracy: 86.6%
```

## Applying Optimal Parameters

Once you find optimal parameters, you can use them in your code:

### Option 1: Modify Default Values

Edit `deepfake_detector/ensemble_detector.py`:

```python
def __init__(self, 
             models: Optional[List[str]] = None,
             weights: Optional[Dict[str, float]] = None,
             threshold: float = 0.52,  # Your optimal value
             confidence_threshold: float = 0.48,  # Your optimal value
             strategy: str = 'weighted_average'):
```

And update weights:

```python
if weights is None:
    weights = {
        'clip_vit': 0.55,      # Your optimal value
        'face_xray': 0.30,     # Your optimal value
        'metadata': 0.10,
        'camera_pipeline': 0.05
    }
```

### Option 2: Use Custom Parameters in Code

```python
from deepfake_detector import DeepfakeDetector, EnsembleDetector

# Create with optimal parameters
ensemble = EnsembleDetector(
    models=['clip_vit', 'face_xray'],
    weights={
        'clip_vit': 0.55,
        'face_xray': 0.30,
        'metadata': 0.10,
        'camera_pipeline': 0.05
    },
    threshold=0.52,
    confidence_threshold=0.48
)

detector = DeepfakeDetector()
detector.image_detector = EnsembleImageDetectorWrapper(ensemble)
```

## Parameter Ranges Tested

The script tests these ranges:

### Thresholds
- 0.45, 0.50, 0.55, 0.60

### Confidence Thresholds
- 0.35, 0.40, 0.45, 0.50, 0.55

### Weight Combinations
- CLIP-ViT: 0.45 - 0.60
- Face X-Ray: 0.20 - 0.30
- Metadata: 0.05 - 0.10
- Camera Pipeline: 0.05

## Tips

1. **More data = better calibration**: Use as many test images as possible
2. **Balance recall vs precision**: 
   - High recall = catch more deepfakes (but more false positives)
   - High precision = fewer false positives (but might miss deepfakes)
3. **Test on representative data**: Use images similar to what you'll detect in production
4. **Re-calibrate periodically**: As new deepfake techniques emerge, recalibrate

## Current Performance

Based on your latest test:
- Deepfakes: 74/100 (74% recall)
- Real images: 44/101 false positives (44% false positive rate)

**Goal**: Find parameters that achieve:
- 80-90% recall (catch 80-90% of deepfakes)
- 10-20% false positive rate (flag 10-20% of real images)

Run the calibration script to find the optimal balance!

