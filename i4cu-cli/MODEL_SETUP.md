# Pre-trained Model Setup Guide

This guide explains how to set up and use pre-trained deepfake detection models.

## 🚀 Quick Start - Modern Models (Recommended)

**Easiest Option**: Use modern models that auto-download from Hugging Face:

```bash
# Install dependencies
pip install transformers torch torchvision

# Use CLIP-ViT (best accuracy, auto-downloads)
python cli.py image.jpg --model clip_vit

# Or use ensemble (combines multiple models)
python cli.py image.jpg --model ensemble
```

**Why modern models?**
- ✅ Auto-download from Hugging Face (no manual downloads!)
- ✅ State-of-the-art accuracy (85-95% vs 60-70%)
- ✅ Works on latest deepfakes (Veo AI, DALL-E, Midjourney, etc.)
- ✅ No need to hunt for model weights

**Available Modern Models:**
- `clip_vit` - CLIP-ViT (recommended, best accuracy)
- `uia_vit` - UIA-ViT (also excellent)
- `face_xray` - Face X-Ray (for face-swap deepfakes)
- `ensemble` - Combines multiple models (best overall)

## Traditional Model Setup

The tool will automatically try to load pre-trained models. If PyTorch is installed, it will attempt to use:
1. XceptionNet (FaceForensics++)
2. MesoNet
3. EfficientNet
4. Heuristic model (fallback)

## Installing PyTorch (Recommended)

For best results, install PyTorch:

```bash
# CPU only
pip install torch torchvision

# With CUDA (for GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Downloading Pre-trained Weights

### Option 1: XceptionNet (FaceForensics++)

1. Download weights from FaceForensics++ repository:
   - Visit: https://github.com/ondyari/FaceForensics
   - Or download directly: https://github.com/ondyari/FaceForensics/releases

2. Place the weights file in:
   ```
   ~/.deepfake_detector/models/xception.pth
   ```

3. Or specify path when loading:
   ```python
   from deepfake_detector import DeepfakeDetector, ModelLoader
   
   loader = ModelLoader()
   model = loader.load_image_model('xception', model_name='path/to/xception.pth')
   detector = DeepfakeDetector(image_ml_model=model)
   ```

### Option 2: MesoNet

1. Download MesoNet weights:
   - Available from various sources (Kaggle, GitHub repos)
   - Look for "MesoNet" or "Meso4" pretrained weights

2. Place in:
   ```
   ~/.deepfake_detector/models/mesonet.pth
   ```

3. Or use directly:
   ```python
   loader = ModelLoader()
   model = loader.load_image_model('mesonet', model_name='path/to/mesonet.pth')
   ```

### Option 3: EfficientNet

EfficientNet will use ImageNet pretrained weights and can be fine-tuned. The tool will attempt to use pretrained EfficientNet-B4.

### Option 4: Using Your Own Model

```python
from deepfake_detector import DeepfakeDetector
import torch

# Load your custom PyTorch model
model = torch.load('your_model.pth')
model.eval()

# Use with detector
detector = DeepfakeDetector(image_ml_model=model)
```

## Model Sources

### Recommended Sources:

1. **FaceForensics++**
   - GitHub: https://github.com/ondyari/FaceForensics
   - Provides XceptionNet weights
   - High accuracy for face-swap deepfakes

2. **DeepfakeBench**
   - Repository with 20+ pretrained models
   - Includes Xception, EfficientNet, FaceXray, etc.

3. **Kaggle DFDC Challenge**
   - Winning models with pretrained weights
   - High accuracy but larger models

4. **MesoNet**
   - Lightweight and fast
   - Good for CPU-only systems
   - Available on various GitHub repos

## Usage Examples

### Using Auto Model Loading (Default)

```python
from deepfake_detector import DeepfakeDetector

# Will automatically try to load best available model
detector = DeepfakeDetector()
result = detector.detect('image.jpg')
```

### Specifying Model Type

```python
# Try XceptionNet first
detector = DeepfakeDetector(model_type='xception')

# Or MesoNet
detector = DeepfakeDetector(model_type='mesonet')

# Or EfficientNet
detector = DeepfakeDetector(model_type='efficientnet')

# Force heuristic (no ML models)
detector = DeepfakeDetector(model_type='heuristic')
```

### Using CLI with Models

```bash
# Auto-load best available model
python cli.py image.jpg

# The tool will automatically use PyTorch models if available
```

## Model Performance

| Model | Accuracy | Speed | GPU Required | Best For |
|-------|----------|-------|--------------|----------|
| XceptionNet | High (~90%+) | Medium | Recommended | Face-swap deepfakes |
| MesoNet | Medium-High (~85%+) | Fast | No | CPU systems, quick detection |
| EfficientNet | High (~90%+) | Medium | Recommended | General deepfakes |
| Heuristic | Low-Medium (~60-70%) | Very Fast | No | Fallback, basic detection |

## Troubleshooting

### "Model not found" warnings

This means pretrained weights aren't available. Options:
1. Download weights and place in `~/.deepfake_detector/models/`
2. Use heuristic model (automatic fallback)
3. Train your own model

### PyTorch not installed

Install PyTorch:
```bash
pip install torch torchvision
```

### CUDA/GPU issues

Models will automatically use CPU if GPU isn't available. For GPU support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Low accuracy

- Ensure you're using pretrained weights (not random initialization)
- Try different model types
- Check that images are properly preprocessed
- Consider using ensemble of multiple models

## Advanced: Training Your Own Model

If you have training data:

1. Use the model architectures provided (MesoNet, Xception, etc.)
2. Train on your dataset
3. Save weights
4. Load with ModelLoader

Example:
```python
from deepfake_detector.model_loader import MesoNet
import torch

# Train your model...
model = MesoNet()
# ... training code ...

# Save
torch.save(model.state_dict(), 'my_mesonet.pth')

# Load later
loader = ModelLoader()
model = loader.load_image_model('mesonet', 'my_mesonet.pth')
```

## Notes

- Models are cached in `~/.deepfake_detector/models/` after first load
- GPU acceleration significantly speeds up inference
- For production, consider using ensemble of multiple models
- Always verify results, especially for high-stakes decisions

