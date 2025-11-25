"""
Wrapper to make EnsembleDetector compatible with ImageDetector interface.
"""

from typing import Dict
from .ensemble_detector import EnsembleDetector


class EnsembleImageDetectorWrapper:
    """Wrapper to make EnsembleDetector work with existing ImageDetector interface."""
    
    def __init__(self, ensemble_detector: EnsembleDetector):
        self.ensemble_detector = ensemble_detector
        self.ml_model = None  # For compatibility
        self.threshold = ensemble_detector.threshold
        self.confidence_threshold = ensemble_detector.confidence_threshold
    
    def detect(self, image_path: str) -> Dict:
        """Detect using ensemble - compatible with ImageDetector interface."""
        result = self.ensemble_detector.detect(image_path)
        
        # Convert to ImageDetector format for compatibility
        return {
            'file_path': result['file_path'],
            'file_type': result['file_type'],
            'exif_analysis': {
                'exif_data': result.get('metadata_details', {}).get('exif_data', {}),
                'suspicious_score': result.get('metadata_score', 0.0),
                'findings': result.get('metadata_details', {}).get('findings', [])
            },
            'ocr_analysis': {
                'text_found': False,
                'text_content': '',
                'suspicious_score': 0.0,
                'findings': []
            },
            'ml_prediction': {
                'score': result['ensemble_score'],
                'confidence': result['confidence'],
                'prediction': 'deepfake' if result['is_deepfake'] else 'real',
                'model_type': 'Ensemble',
                'model_scores': result.get('model_scores', {}),
                'metadata_score': result.get('metadata_score', 0.0),
                'camera_pipeline_score': result.get('camera_pipeline_score', 0.0)
            },
            'overall_score': result['ensemble_score'],
            'is_deepfake': result['is_deepfake'],
            'confidence': result['confidence'],
            'warnings': result.get('warnings', [])
        }

