"""
Deepfake Detector Tool
A comprehensive tool for detecting deepfakes in images, videos, and audio
using EXIF data analysis, OCR, and ML models.
"""

__version__ = "1.0.0"

from .detector import DeepfakeDetector
from .image_detector import ImageDetector
from .video_detector import VideoDetector
from .audio_detector import AudioDetector
from .ensemble_detector import EnsembleDetector
from .model_loader import ModelLoader, load_default_image_model, HeuristicImageModel

__all__ = [
    'DeepfakeDetector',
    'ImageDetector',
    'VideoDetector',
    'AudioDetector',
    'EnsembleDetector',
    'ModelLoader',
    'load_default_image_model',
    'HeuristicImageModel',
]

