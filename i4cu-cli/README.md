# Deepfake Detector Tool

A comprehensive tool for detecting deepfakes in images, videos, and audio files using multiple detection techniques including EXIF data analysis, OCR, and machine learning models.

## Features

- **Multi-format Support**: Detects deepfakes in images (JPG, PNG, etc.), videos (MP4, AVI, etc.), and audio files (MP3, WAV, etc.)
- **EXIF Data Analysis**: Analyzes image metadata for suspicious patterns and manipulation indicators
- **OCR Analysis**: Detects text artifacts and watermarks that might indicate deepfake generation
- **Video Frame Analysis**: Extracts and analyzes frames from videos for comprehensive detection
- **Audio Feature Analysis**: Analyzes spectral features and audio characteristics
- **ML Model Integration**: Supports integration with pre-trained deepfake detection models
- **Batch Processing**: Process multiple files or entire directories at once
- **Detailed Reporting**: Provides comprehensive analysis reports with confidence scores

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR** (required for OCR analysis):
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

## Usage

### Command Line Interface

#### Detect a single file:
```bash
python cli.py image.jpg
python cli.py video.mp4
python cli.py audio.mp3
```

#### Detect all files in a directory:
```bash
python cli.py --directory testing-data
python cli.py -d testing-data --recursive
```

#### Verbose output with detailed analysis:
```bash
python cli.py image.jpg --verbose
```

#### Save results to JSON:
```bash
python cli.py image.jpg --output results.json
```

#### JSON-only output:
```bash
python cli.py image.jpg --json
```

### Python API

```python
from deepfake_detector import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector()

# Detect a single file
result = detector.detect('image.jpg')
print(f"Is deepfake: {result['is_deepfake']}")
print(f"Confidence: {result['confidence']:.2%}")

# Detect multiple files
results = detector.detect_batch(['image1.jpg', 'image2.jpg'])

# Detect all files in a directory
results = detector.detect_directory('testing-data', recursive=True)
```

## Detection Methods

### Image Detection

1. **EXIF Analysis**: 
   - Checks for suspicious software metadata
   - Identifies missing or inconsistent timestamps
   - Detects unusual camera models or metadata patterns

2. **OCR Analysis**:
   - Extracts text from images
   - Detects suspicious keywords or watermarks
   - Identifies text artifacts that might indicate manipulation

3. **ML Model Prediction**:
   - Uses pre-trained deepfake detection models (if available)
   - Analyzes image features for deepfake characteristics

### Video Detection

- Extracts frames at regular intervals (default: 1 frame per second)
- Applies image detection techniques to each frame
- Aggregates results to determine overall video score
- Provides frame-by-frame analysis statistics

### Audio Detection

- Analyzes spectral features (MFCC, spectral centroid, etc.)
- Detects unusual patterns that might indicate voice cloning
- Checks for abrupt transitions or splicing artifacts
- Uses ML models for audio deepfake detection (if available)

## Output Format

The tool provides detailed results including:

- **Overall Score**: 0.0 (real) to 1.0 (deepfake)
- **Confidence**: How confident the detection is
- **Detailed Analysis**: EXIF data, OCR results, feature analysis
- **Warnings**: Any suspicious findings or errors
- **Statistics**: For videos, frame-by-frame statistics

## ML Model Integration

The tool now includes support for pre-trained deepfake detection models:

### Automatic Model Loading (Default)

```python
from deepfake_detector import DeepfakeDetector

# Automatically tries to load best available model
detector = DeepfakeDetector()
```

### Using Specific Models

```python
# Use XceptionNet (FaceForensics++)
detector = DeepfakeDetector(model_type='xception')

# Use MesoNet (lightweight)
detector = DeepfakeDetector(model_type='mesonet')

# Use EfficientNet
detector = DeepfakeDetector(model_type='efficientnet')

# Force heuristic model
detector = DeepfakeDetector(model_type='heuristic')
```

### CLI Model Selection

```bash
# Use specific model
python cli.py image.jpg --model xception
python cli.py image.jpg --model mesonet
```

### Downloading Pre-trained Models

See `MODEL_SETUP.md` for detailed instructions on downloading and setting up pre-trained models.

Quick setup:
```bash
# Install PyTorch (required for ML models)
pip install torch torchvision

# Check available models
python download_models.py --list

# The tool will automatically use models if available
```

### Custom Models

```python
from deepfake_detector import DeepfakeDetector
import torch

# Load your custom PyTorch model
model = torch.load('your_model.pth')
model.eval()

# Use with detector
detector = DeepfakeDetector(image_ml_model=model)
```

## Testing

The repository includes test data in the `testing-data` directory:
- `deepfake/`: Deepfake images
- `real/`: Real images
- `pics/`: Additional test images
- Various video files (`.mp4`)

Run tests:
```bash
python cli.py --directory testing-data --verbose
```

## Limitations

- ML models are not included by default - you need to provide your own trained models
- OCR requires Tesseract to be installed separately
- Video processing can be slow for large files
- Detection accuracy depends on the quality of the input and the ML models used

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is provided as-is for educational and research purposes.

## Project Structure

```
i4cu-cli/
├── i4cu-deepfake_detector/     # Main package
│   ├── __init__.py            # Package initialization
│   ├── detector.py            # Main detector orchestrator
│   ├── image_detector.py      # Image deepfake detection
│   ├── video_detector.py      # Video deepfake detection
│   ├── audio_detector.py      # Audio deepfake detection
│   ├── ensemble_detector.py   # Ensemble detection
│   ├── model_loader.py        # ML model loading
│   └── ...
├── cli.py                     # Command-line interface
├── example_usage.py           # Example usage scripts
├── calibrate_ensemble.py      # Calibration tool
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── QUICKSTART.md              # Quick start guide
├── MODEL_SETUP.md             # Model setup guide
├── ENSEMBLE_GUIDE.md          # Ensemble detection guide
├── HOW_SCORING_WORKS.md       # Scoring explanation
└── testing-data/              # Test data directory
```

## Notes

- The tool uses heuristic methods and ML models (if provided) to detect deepfakes
- No detection method is 100% accurate
- Always verify results with multiple methods and expert analysis
- Keep your ML models updated for best results

