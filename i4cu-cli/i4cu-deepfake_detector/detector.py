"""
Main Deepfake Detector
Orchestrates detection across images, videos, and audio.
"""

import os
from typing import Dict, List, Optional, Union
from pathlib import Path
from .image_detector import ImageDetector
from .video_detector import VideoDetector
from .audio_detector import AudioDetector
from .audio_ensemble_detector import AudioEnsembleDetector
from .ensemble_detector import EnsembleDetector
from .ensemble_wrapper import EnsembleImageDetectorWrapper


class DeepfakeDetector:
    """Main class for deepfake detection across multiple media types."""
    
    # Supported file extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    
    def __init__(self, image_ml_model=None, audio_ml_model=None, model_type: str = 'auto',
                 threshold: float = 0.52, confidence_threshold: float = 0.5,
                 use_ensemble: bool = False, ensemble_models: Optional[List[str]] = None):
        """
        Initialize the deepfake detector.
        
        Args:
            image_ml_model: Optional ML model for image deepfake detection
            audio_ml_model: Optional ML model for audio deepfake detection
            model_type: Type of model to auto-load if image_ml_model is None 
                       ('auto', 'clip_vit', 'uia_vit', 'face_xray', 'xception', 'mesonet', 'efficientnet', 'heuristic', 'ensemble')
            threshold: Detection threshold (default: 0.52, conservative to minimize false positives)
            confidence_threshold: Minimum confidence required (default: 0.5, higher to reduce false positives)
            use_ensemble: Use ensemble detection (combines multiple models) - default: False
            ensemble_models: List of models to use in ensemble ['clip_vit', 'uia_vit', 'face_xray']
        """
        self.use_ensemble = use_ensemble or (model_type == 'ensemble')
        
        if self.use_ensemble:
            # Use ensemble detector with conservative thresholds to reduce false positives
            self.ensemble_detector = EnsembleDetector(
                models=ensemble_models,
                threshold=max(threshold, 0.52),  # Minimum 0.52 for ensemble (raised to reduce false positives)
                confidence_threshold=max(confidence_threshold, 0.48)  # Minimum 0.48 for ensemble
            )
            # Create a wrapper image detector for compatibility
            self.image_detector = EnsembleImageDetectorWrapper(self.ensemble_detector)
            # Use audio ensemble detector for audio files
            self.audio_detector = AudioEnsembleDetector()
        else:
            # Use single model
            if image_ml_model is None:
                from .model_loader import load_default_image_model
                image_ml_model = load_default_image_model(model_type)
            
            self.image_detector = ImageDetector(ml_model=image_ml_model, 
                                               threshold=threshold,
                                               confidence_threshold=confidence_threshold)

            # Simple audio detector (no ensemble) for non-ensemble mode
            self.audio_detector = AudioDetector(ml_model=audio_ml_model)

        self.video_detector = VideoDetector(image_detector=self.image_detector)
    
    def detect(self, file_path: Union[str, Path]) -> Dict:
        """
        Detect deepfakes in a file (image, video, or audio).
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary containing detection results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                'error': f"File not found: {file_path}",
                'file_path': str(file_path)
            }
        
        extension = file_path.suffix.lower()
        
        if extension in self.IMAGE_EXTENSIONS:
            return self.image_detector.detect(str(file_path))
        elif extension in self.VIDEO_EXTENSIONS:
            return self.video_detector.detect(str(file_path))
        elif extension in self.AUDIO_EXTENSIONS:
            return self.audio_detector.detect(str(file_path))
        else:
            return {
                'error': f"Unsupported file type: {extension}",
                'file_path': str(file_path),
                'supported_types': {
                    'images': list(self.IMAGE_EXTENSIONS),
                    'videos': list(self.VIDEO_EXTENSIONS),
                    'audio': list(self.AUDIO_EXTENSIONS)
                }
            }
    
    def detect_batch(self, file_paths: List[Union[str, Path]], 
                     show_progress: bool = True) -> List[Dict]:
        """
        Detect deepfakes in multiple files.
        
        Args:
            file_paths: List of file paths to analyze
            show_progress: Whether to show progress (if tqdm is available)
            
        Returns:
            List of detection result dictionaries
        """
        results = []
        
        try:
            from tqdm import tqdm
            iterator = tqdm(file_paths, desc="Processing files") if show_progress else file_paths
        except ImportError:
            iterator = file_paths
        
        for file_path in iterator:
            result = self.detect(file_path)
            results.append(result)
        
        return results
    
    def detect_directory(self, directory_path: Union[str, Path],
                         recursive: bool = True) -> List[Dict]:
        """
        Detect deepfakes in all supported files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
            
        Returns:
            List of detection result dictionaries
        """
        directory_path = Path(directory_path)
        
        if not directory_path.is_dir():
            return [{
                'error': f"Not a directory: {directory_path}",
                'file_path': str(directory_path)
            }]
        
        all_extensions = self.IMAGE_EXTENSIONS | self.VIDEO_EXTENSIONS | self.AUDIO_EXTENSIONS
        
        file_paths = []
        if recursive:
            for ext in all_extensions:
                file_paths.extend(directory_path.rglob(f"*{ext}"))
                file_paths.extend(directory_path.rglob(f"*{ext.upper()}"))
        else:
            for ext in all_extensions:
                file_paths.extend(directory_path.glob(f"*{ext}"))
                file_paths.extend(directory_path.glob(f"*{ext.upper()}"))
        
        return self.detect_batch(file_paths)

