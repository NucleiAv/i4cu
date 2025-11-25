#!/usr/bin/env python3
"""Debug script to test ensemble detection and see what's happening."""

import sys
from pathlib import Path
from deepfake_detector import EnsembleDetector

# Test on a single image
test_image = sys.argv[1] if len(sys.argv) > 1 else "testing-data/deepfake/df00001.jpg"

print(f"Testing ensemble detection on: {test_image}")
print("=" * 60)

# Create ensemble detector
ensemble = EnsembleDetector(
    models=['clip_vit', 'uia_vit', 'face_xray'],
    threshold=0.45,
    confidence_threshold=0.35
)

# Check which models loaded
print(f"\nLoaded models: {list(ensemble.ml_models.keys())}")
print(f"Model count: {len(ensemble.ml_models)}")

if len(ensemble.ml_models) == 0:
    print("ERROR: No models loaded!")
    sys.exit(1)

# Run detection
result = ensemble.detect(test_image)

print(f"\nDetection Results:")
print(f"  File: {result['file_path']}")
print(f"  Model Scores: {result.get('model_scores', {})}")
print(f"  Metadata Score: {result.get('metadata_score', 0):.3f}")
print(f"  Camera Pipeline Score: {result.get('camera_pipeline_score', 0):.3f}")
print(f"  Ensemble Score: {result.get('ensemble_score', 0):.3f}")
print(f"  Confidence: {result.get('confidence', 0):.3f}")
print(f"  Is Deepfake: {result.get('is_deepfake', False)}")
print(f"  Threshold: {ensemble.threshold}")
print(f"  Confidence Threshold: {ensemble.confidence_threshold}")

if result.get('warnings'):
    print(f"\nWarnings: {result['warnings']}")

if result.get('error'):
    print(f"\nError: {result['error']}")

