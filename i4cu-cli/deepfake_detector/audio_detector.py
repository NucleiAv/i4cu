"""
Audio Deepfake Detector
Analyzes audio files for deepfake characteristics.
"""

import os
import numpy as np
from typing import Dict, List, Optional
import librosa
import soundfile as sf
from pathlib import Path


class AudioDetector:
    """Detects deepfakes in audio files using various audio analysis techniques."""
    
    def __init__(self, ml_model=None):
        """
        Initialize the audio detector.
        
        Args:
            ml_model: Optional pre-trained ML model for audio deepfake detection
        """
        self.ml_model = ml_model
    
    def detect(self, audio_path: str) -> Dict:
        """
        Perform comprehensive deepfake detection on an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing detection results and scores
        """
        results = {
            'file_path': audio_path,
            'file_type': 'audio',
            'audio_info': {},
            'feature_analysis': {},
            'ml_prediction': {},
            'overall_score': 0.0,
            'is_deepfake': False,
            'confidence': 0.0,
            'warnings': []
        }
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Get audio information
            audio_info = self._get_audio_info(audio_path, y, sr)
            results['audio_info'] = audio_info
            
            # Feature analysis
            feature_results = self._analyze_features(y, sr)
            results['feature_analysis'] = feature_results
            
            # ML model prediction
            if self.ml_model:
                ml_results = self._ml_predict(y, sr)
                results['ml_prediction'] = ml_results
            else:
                results['ml_prediction'] = {
                    'score': 0.5,
                    'note': 'No ML model loaded'
                }
            
            # Calculate overall score
            results['overall_score'], results['confidence'] = self._calculate_score(
                feature_results, results['ml_prediction']
            )
            
            # Determine if deepfake
            results['is_deepfake'] = results['overall_score'] > 0.6
        
        except Exception as e:
            results['error'] = str(e)
            results['warnings'].append(f"Error processing audio: {str(e)}")
        
        return results
    
    def _get_audio_info(self, audio_path: str, y: np.ndarray, sr: int) -> Dict:
        """Extract audio file metadata and information."""
        info = {
            'sample_rate': int(sr),
            'duration_seconds': len(y) / sr,
            'channels': 1 if len(y.shape) == 1 else y.shape[1],
            'file_size_bytes': os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        }
        
        try:
            # Try to get metadata using soundfile
            with sf.SoundFile(audio_path) as f:
                info['subtype'] = f.subtype
                info['format'] = f.format
                info['sections'] = f.sections
        except:
            pass
        
        return info
    
    def _analyze_features(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze audio features for deepfake indicators."""
        features = {
            'suspicious_score': 0.0,
            'findings': [],
            'feature_values': {}
        }
        
        try:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            features['feature_values'] = {
                'mean_spectral_centroid': float(np.mean(spectral_centroids)),
                'std_spectral_centroid': float(np.std(spectral_centroids)),
                'mean_spectral_rolloff': float(np.mean(spectral_rolloff)),
                'mean_zero_crossing_rate': float(np.mean(zero_crossing_rate)),
                'mean_mfcc': [float(x) for x in np.mean(mfccs, axis=1)]
            }
            
            # Check for unusual patterns that might indicate deepfake
            # High variance in spectral features might indicate artifacts
            if np.std(spectral_centroids) > np.mean(spectral_centroids) * 0.5:
                features['suspicious_score'] += 0.2
                features['findings'].append("High variance in spectral centroid (possible artifacts)")
            
            # Unusual zero crossing rate
            if np.mean(zero_crossing_rate) < 0.01 or np.mean(zero_crossing_rate) > 0.3:
                features['suspicious_score'] += 0.15
                features['findings'].append("Unusual zero crossing rate")
            
            # Check for silence or very low energy (might indicate manipulation)
            rms = librosa.feature.rms(y=y)[0]
            if np.mean(rms) < 0.01:
                features['suspicious_score'] += 0.1
                features['findings'].append("Very low energy detected")
            
            # Check for abrupt transitions (might indicate splicing)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) > len(y) / sr * 10:  # More than 10 onsets per second
                features['suspicious_score'] += 0.15
                features['findings'].append("Unusually high number of abrupt transitions")
        
        except Exception as e:
            features['findings'].append(f"Error in feature analysis: {str(e)}")
        
        features['suspicious_score'] = min(features['suspicious_score'], 1.0)
        return features
    
    def _ml_predict(self, y: np.ndarray, sr: int) -> Dict:
        """Use ML model to predict if audio is a deepfake."""
        if self.ml_model is None:
            return {'score': 0.5, 'confidence': 0.0, 'note': 'No ML model available'}
        
        try:
            # Extract features for model
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features = np.mean(mfccs, axis=1)
            features = features.reshape(1, -1)
            
            # Predict (placeholder - actual implementation depends on model)
            if hasattr(self.ml_model, 'predict'):
                prediction = self.ml_model.predict(features)
                if isinstance(prediction, np.ndarray):
                    score = float(prediction[0][0] if prediction.shape[1] > 1 else prediction[0])
                else:
                    score = float(prediction)
            else:
                score = 0.5
            
            return {
                'score': score,
                'confidence': abs(score - 0.5) * 2,
                'prediction': 'deepfake' if score > 0.5 else 'real'
            }
        
        except Exception as e:
            return {
                'score': 0.5,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_score(self, feature_results: Dict, ml_results: Dict) -> tuple:
        """Calculate overall deepfake score from all analyses."""
        weights = {
            'features': 0.4,
            'ml': 0.6
        }
        
        feature_score = feature_results.get('suspicious_score', 0.0)
        ml_score = ml_results.get('score', 0.5)
        
        # Normalize ML score
        ml_normalized = abs(ml_score - 0.5) * 2 if ml_score > 0.5 else 0.0
        
        overall_score = (
            feature_score * weights['features'] +
            ml_normalized * weights['ml']
        )
        
        confidence = min(
            max(feature_score, ml_normalized) * 1.2,
            1.0
        )
        
        return min(overall_score, 1.0), confidence

