# Quick Start Guide

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR** (required for OCR analysis):
   - **Windows**: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

3. **Install ML Models (Optional but Recommended)**:
   ```bash
   # For best accuracy, install PyTorch and transformers
   pip install transformers torch torchvision
   ```
   Models will auto-download from Hugging Face on first use!

## Basic Usage

### Command Line

**Detect a single image:**
```bash
python cli.py testing-data/real/45_288.jpg
```

**Detect a video:**
```bash
python cli.py testing-data/deevid-2.mp4
```

**Detect all files in a directory:**
```bash
python cli.py --directory testing-data
```

**Get detailed analysis:**
```bash
python cli.py testing-data/real/45_288.jpg --verbose
```

**Save results to JSON:**
```bash
python cli.py testing-data/real/45_288.jpg --output result.json
```

**Use specific ML model (recommended for accuracy):**
```bash
# Best accuracy - CLIP-ViT (auto-downloads from Hugging Face)
python cli.py image.jpg --model clip_vit

# Also excellent - UIA-ViT
python cli.py image.jpg --model uia_vit

# For face-swap deepfakes
python cli.py image.jpg --model face_xray

# Use ensemble (combines multiple models - best accuracy)
python cli.py image.jpg --model ensemble
```

### Python API

```python
from deepfake_detector import DeepfakeDetector

# Initialize (auto-selects best available model)
detector = DeepfakeDetector()

# Or specify model type
detector = DeepfakeDetector(model_type='clip_vit')  # Best accuracy
detector = DeepfakeDetector(model_type='ensemble')   # Best overall

# Detect single file
result = detector.detect('testing-data/real/45_288.jpg')
print(f"Is deepfake: {result['is_deepfake']}")
print(f"Score: {result['overall_score']:.2%}")

# Batch processing
results = detector.detect_batch(['file1.jpg', 'file2.jpg'])

# Directory processing
results = detector.detect_directory('testing-data')
```

## Modern Models (2024-2025)

For best results with modern deepfakes (Veo AI, DALL-E, etc.), use:

```bash
# Install modern models support
pip install transformers torch torchvision

# Use CLIP-ViT (recommended - 85-95% accuracy)
python cli.py video.mp4 --model clip_vit --verbose

# Or use ensemble (combines multiple models)
python cli.py video.mp4 --model ensemble --verbose
```

**Why modern models?**
- ✅ Auto-download from Hugging Face (no manual weight downloads!)
- ✅ State-of-the-art accuracy (85-95% vs 60-70% with heuristics)
- ✅ Works on latest deepfakes (Veo, DALL-E, Midjourney, etc.)
- ✅ Much better than heuristic-only detection

## Testing with Your Data

The tool includes test data in the `testing-data` directory:
- `real/` - Real images
- `deepfake/` - Deepfake images  
- `pics/` - Additional test images
- Various `.mp4` video files

Run a test:
```bash
python cli.py --directory testing-data --verbose
```

## Understanding Results

- **Score**: 0.0 (real) to 1.0 (deepfake)
- **Confidence**: How confident the detection is (0-1)
- **is_deepfake**: Boolean indicating if deepfake was detected
- **Warnings**: List of suspicious findings

**Score Ranges:**
- 0.0 - 0.45: Likely Real ✅
- 0.45 - 0.55: Uncertain
- 0.55 - 1.0: Deepfake Detected 🔴

## Next Steps

- See `README.md` for full documentation
- See `MODEL_SETUP.md` for detailed model setup
- See `ENSEMBLE_GUIDE.md` for ensemble detection
- Check `example_usage.py` for more examples

