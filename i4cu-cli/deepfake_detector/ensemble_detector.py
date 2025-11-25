"""
Ensemble Deepfake Detector
Combines multiple models and analysis methods for improved accuracy.
"""

import warnings
import logging

# Suppress NNPACK warnings (harmless - PyTorch falls back to default CPU implementation)
warnings.filterwarnings('ignore', message='.*NNPACK.*')
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image
from .model_loader import ModelLoader, load_default_image_model
from .image_detector import ImageDetector


class EnsembleDetector:
    """
    Ensemble detector that combines multiple models and analysis methods.
    Uses weighted voting/averaging for final decision.
    """
    
    def __init__(self, 
                 models: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 threshold: float = 0.52,  # Raised to reduce false positives (from 0.50)
                 confidence_threshold: float = 0.48,  # Raised to require stronger evidence (from 0.45)
                 strategy: str = 'weighted_average'):
        """
        Initialize ensemble detector.
        
        Args:
            models: List of model types to use ['clip_vit', 'uia_vit', 'face_xray']
                   If None, uses all available
            weights: Custom weights for each model (must sum to 1.0)
            threshold: Detection threshold
            confidence_threshold: Minimum confidence required
            strategy: Combination strategy ('weighted_average', 'majority_vote', 'max', 'consensus')
        """
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        self.strategy = strategy
        
        # Default models to use
        # Note: UIA-ViT is a general model, not deepfake-specific, so it may give unreliable scores
        if models is None:
            models = ['clip_vit', 'face_xray']  # Removed uia_vit by default - use only if you have a trained model
        self.models = models
        
        # Default weights (can be customized)
        # UIA-ViT weight reduced because it's using a general model, not deepfake-specific
        if weights is None:
            weights = {
                'clip_vit': 0.50,      # CLIP-ViT - excellent for inconsistencies (increased weight)
                'uia_vit': 0.10,       # UIA-ViT - reduced weight (general model, not deepfake-specific)
                'face_xray': 0.25,     # Face X-Ray - compositing artifacts (increased weight)
                'metadata': 0.10,      # Metadata integrity
                'camera_pipeline': 0.05  # Camera-pipeline consistency
            }
        self.weights = weights
        
        # Load models
        self.loader = ModelLoader()
        self.ml_models = {}
        self._load_models()
        
        # Create individual detector for metadata/OCR analysis
        # We'll create a minimal one just for EXIF/OCR
        try:
            from .image_detector import ImageDetector
            self.metadata_detector = ImageDetector(threshold=0.0, confidence_threshold=0.0)
        except:
            self.metadata_detector = None
    
    def _load_models(self):
        """Load all specified models."""
        for model_type in self.models:
            try:
                model = self.loader.load_image_model(model_type)
                if model is not None:
                    self.ml_models[model_type] = model
            except Exception as e:
                # Suppress warnings during model loading
                pass
        
        # If no models loaded, fall back to heuristic
        if len(self.ml_models) == 0:
            try:
                heuristic_model = self.loader.load_image_model('heuristic')
                if heuristic_model:
                    self.ml_models['heuristic'] = heuristic_model
            except:
                pass
    
    def detect(self, image_path: str) -> Dict:
        """
        Perform ensemble detection on an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing ensemble detection results
        """
        results = {
            'file_path': image_path,
            'file_type': 'image',
            'model_scores': {},
            'metadata_score': 0.0,
            'camera_pipeline_score': 0.0,
            'ensemble_score': 0.0,
            'confidence': 0.0,
            'is_deepfake': False,
            'warnings': []
        }
        
        try:
            # Load image
            img = Image.open(image_path)
            
            # Run all ML models
            model_scores = {}
            
            # If no models loaded, this is a critical issue
            if len(self.ml_models) == 0:
                results['warnings'].append("No ML models loaded! Falling back to heuristic.")
                # Try to use ImageDetector as fallback
                if self.metadata_detector and hasattr(self.metadata_detector, 'ml_model'):
                    try:
                        fallback_result = self.metadata_detector._ml_predict(img)
                        model_scores['fallback'] = fallback_result.get('score', 0.5)
                    except:
                        model_scores['fallback'] = 0.5
                else:
                    model_scores['heuristic'] = 0.5  # Default neutral
            
            for model_type, model in self.ml_models.items():
                try:
                    # Preprocess for model
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized)
                    if img_array.max() > 1.0:
                        img_array = img_array.astype(np.float32) / 255.0
                    
                    # Get prediction
                    if hasattr(model, 'predict'):
                        score = model.predict(img_array)
                    elif callable(model):
                        score = model(img_array)
                    else:
                        score = 0.5  # Fallback
                    
                    # Ensure score is in [0, 1] range
                    score = float(score)
                    if score < 0:
                        score = 0.0
                    elif score > 1:
                        score = 1.0
                    
                    model_scores[model_type] = score
                except Exception as e:
                    results['warnings'].append(f"Error with {model_type}: {str(e)}")
                    model_scores[model_type] = 0.5  # Neutral score on error
            
            results['model_scores'] = model_scores
            
            # Get metadata analysis
            metadata_results = self._analyze_metadata(img, image_path)
            results['metadata_score'] = metadata_results['score']
            results['metadata_details'] = metadata_results
            
            # Get camera-pipeline consistency analysis
            camera_results = self._analyze_camera_pipeline(img, image_path)
            results['camera_pipeline_score'] = camera_results['score']
            results['camera_pipeline_details'] = camera_results
            
            # Combine all scores using ensemble strategy
            ensemble_score, confidence = self._combine_scores(
                model_scores,
                results['metadata_score'],
                results['camera_pipeline_score']
            )
            
            results['ensemble_score'] = ensemble_score
            results['confidence'] = confidence
            
            # Determine if deepfake
            # Use >= instead of > to catch scores exactly at threshold
            # Also allow detection if ensemble_score is high enough even with lower confidence
            # Be more aggressive: if score > 0.5, likely fake
            # Check if any individual model strongly suggests fake
            clip_vit_score = model_scores.get('clip_vit', 0)
            face_xray_score = model_scores.get('face_xray', 0)
            
            # Strong fake indicators - require higher thresholds to reduce false positives
            # Based on calibration: need to reduce 44% false positives to 15-25%
            strong_fake_indicators = [
                clip_vit_score >= 0.68,  # CLIP-ViT >= 68% suggests fake (raised from 0.65)
                face_xray_score >= 0.68,  # Face X-Ray >= 68% suggests fake (raised from 0.65)
                (clip_vit_score >= 0.62 and face_xray_score >= 0.58),  # Both models strongly agree
            ]
            
            # Very strong indicators - require very high scores
            very_strong_indicators = [
                clip_vit_score >= 0.78,  # CLIP-ViT >= 78% is very strong (raised from 0.75)
                (clip_vit_score >= 0.68 and ensemble_score >= 0.52),  # CLIP-ViT high with high ensemble
            ]
            
            # More conservative detection logic to reduce false positives
            # Primary goal: Reduce false positives from 44% to 15-25% while maintaining 80-85% recall
            results['is_deepfake'] = (
                # Primary condition: High ensemble score with high confidence
                (ensemble_score >= self.threshold and confidence >= self.confidence_threshold) or
                # Very high score = definitely fake (but require decent confidence)
                (ensemble_score >= 0.68 and confidence >= 0.45) or
                # High score with strong model indicators (require high confidence)
                (ensemble_score >= 0.58 and confidence >= 0.52 and any(strong_fake_indicators)) or
                # Very strong model indicators (but still require good ensemble and confidence)
                (any(very_strong_indicators) and ensemble_score >= 0.48 and confidence >= 0.42)
            )
            
        except Exception as e:
            results['error'] = str(e)
            results['warnings'].append(f"Error processing image: {str(e)}")
        
        return results
    
    def _analyze_metadata(self, img: Image.Image, image_path: str) -> Dict:
        """Analyze metadata integrity."""
        score = 0.0
        findings = []
        exif_data = {}
        
        try:
            # Use existing EXIF analysis if available
            exif_score = 0.0
            if self.metadata_detector:
                exif_results = self.metadata_detector._analyze_exif(img)
                exif_score = exif_results.get('suspicious_score', 0.0)
                exif_data = exif_results.get('exif_data', {})
            else:
                # Manual EXIF analysis
                try:
                    from PIL.ExifTags import TAGS
                    exif = img._getexif()
                    if exif:
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            if tag in ['Software', 'Artist', 'Copyright', 'Make', 'Model', 
                                      'DateTime', 'DateTimeOriginal']:
                                exif_data[tag] = str(value)
                except:
                    pass
            
            # Check for missing critical metadata
            critical_fields = ['DateTime', 'DateTimeOriginal', 'Make', 'Model']
            missing_fields = [f for f in critical_fields if f not in exif_data]
            
            if len(missing_fields) >= 2:
                score += 0.2
                findings.append(f"Missing critical metadata: {', '.join(missing_fields)}")
            
            # Check for inconsistent timestamps
            if 'DateTime' in exif_data and 'DateTimeOriginal' in exif_data:
                if exif_data['DateTime'] != exif_data['DateTimeOriginal']:
                    score += 0.15
                    findings.append("Inconsistent timestamps detected")
            
            # Check for suspicious software
            if 'Software' in exif_data:
                software = str(exif_data['Software']).lower()
                suspicious = ['deepfake', 'faceswap', 'fake', 'generated', 'ai']
                if any(s in software for s in suspicious):
                    score += 0.3
                    findings.append(f"Suspicious software: {exif_data['Software']}")
            
            # Combine with EXIF suspicious score
            if self.metadata_detector:
                score = max(score, exif_score)
            
        except Exception as e:
            findings.append(f"Metadata analysis error: {str(e)}")
        
        return {
            'score': min(score, 1.0),
            'findings': findings,
            'exif_data': exif_data
        }
    
    def _analyze_camera_pipeline(self, img: Image.Image, image_path: str) -> Dict:
        """Analyze camera-pipeline consistency."""
        score = 0.0
        findings = []
        exif_data = {}
        
        try:
            # Get EXIF data
            if self.metadata_detector:
                exif_results = self.metadata_detector._analyze_exif(img)
                exif_data = exif_results.get('exif_data', {})
            else:
                # Manual EXIF extraction
                try:
                    from PIL.ExifTags import TAGS
                    exif = img._getexif()
                    if exif:
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            if tag in ['Make', 'Model', 'XResolution', 'YResolution']:
                                exif_data[tag] = str(value)
                except:
                    pass
            
            # Check for camera model consistency
            if 'Make' in exif_data and 'Model' in exif_data:
                make_model = f"{exif_data['Make']} {exif_data['Model']}".lower()
                
                # Check for virtual/unknown cameras
                if 'unknown' in make_model or 'virtual' in make_model:
                    score += 0.25
                    findings.append("Unusual camera model detected")
                
                # Check for known camera brands (real cameras)
                known_brands = ['canon', 'nikon', 'sony', 'fujifilm', 'olympus', 'panasonic', 
                              'leica', 'pentax', 'samsung', 'apple', 'huawei', 'xiaomi']
                if not any(brand in make_model for brand in known_brands):
                    if 'make' in exif_data and exif_data['Make']:
                        score += 0.15
                        findings.append("Unrecognized camera brand")
            
            # Check for image dimensions vs camera capabilities
            width, height = img.size
            
            # Check for unusual aspect ratios (might indicate manipulation)
            aspect_ratio = width / height if height > 0 else 1.0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                score += 0.1
                findings.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
            
            # Check for resolution consistency
            if 'XResolution' in exif_data and 'YResolution' in exif_data:
                x_res = float(exif_data['XResolution']) if exif_data['XResolution'] else 72
                y_res = float(exif_data['YResolution']) if exif_data['YResolution'] else 72
                
                if abs(x_res - y_res) > 10:  # Significant difference
                    score += 0.1
                    findings.append("Inconsistent resolution metadata")
            
            # Check for color space consistency
            if img.mode not in ['RGB', 'RGBA', 'L']:
                score += 0.1
                findings.append(f"Unusual color mode: {img.mode}")
            
        except Exception as e:
            findings.append(f"Camera pipeline analysis error: {str(e)}")
        
        return {
            'score': min(score, 1.0),
            'findings': findings
        }
    
    def _combine_scores(self, 
                       model_scores: Dict[str, float],
                       metadata_score: float,
                       camera_pipeline_score: float) -> Tuple[float, float]:
        """
        Combine scores from all models and analyses.
        
        Args:
            model_scores: Dictionary of model_type -> score
            metadata_score: Metadata integrity score
            camera_pipeline_score: Camera-pipeline consistency score
            
        Returns:
            Tuple of (ensemble_score, confidence)
        """
        if self.strategy == 'weighted_average':
            return self._weighted_average(model_scores, metadata_score, camera_pipeline_score)
        elif self.strategy == 'majority_vote':
            return self._majority_vote(model_scores, metadata_score, camera_pipeline_score)
        elif self.strategy == 'max':
            return self._max_vote(model_scores, metadata_score, camera_pipeline_score)
        elif self.strategy == 'consensus':
            return self._consensus(model_scores, metadata_score, camera_pipeline_score)
        else:
            return self._weighted_average(model_scores, metadata_score, camera_pipeline_score)
    
    def _weighted_average(self, 
                         model_scores: Dict[str, float],
                         metadata_score: float,
                         camera_pipeline_score: float) -> Tuple[float, float]:
        """Weighted average combination with outlier handling."""
        total_score = 0.0
        total_weight = 0.0
        
        # Filter out models with extremely low scores (< 0.01) as they're likely not working correctly
        # This handles cases where UIA-ViT (general model) gives useless scores
        filtered_scores = {}
        for model_type, score in model_scores.items():
            # If score is extremely low (< 0.01), it's likely inverted or wrong
            # Only exclude if it's way too low (likely a general model, not deepfake-specific)
            if score < 0.01 and model_type == 'uia_vit':
                # UIA-ViT with very low scores is likely wrong - reduce weight significantly
                weight = self.weights.get(model_type, 0.0) * 0.1  # Reduce to 10% of original weight
            else:
                weight = self.weights.get(model_type, 0.0)
            
            if weight > 0:
                filtered_scores[model_type] = score
                total_score += score * weight
                total_weight += weight
        
        # Add metadata score
        metadata_weight = self.weights.get('metadata', 0.0)
        total_score += metadata_score * metadata_weight
        total_weight += metadata_weight
        
        # Add camera pipeline score
        camera_weight = self.weights.get('camera_pipeline', 0.0)
        total_score += camera_pipeline_score * camera_weight
        total_weight += camera_weight
        
        # Normalize
        if total_weight > 0:
            ensemble_score = total_score / total_weight
        else:
            # If no weights, use simple average of available scores
            all_scores = list(model_scores.values()) + [metadata_score, camera_pipeline_score]
            if len(all_scores) > 0:
                ensemble_score = sum(all_scores) / len(all_scores)
            else:
                ensemble_score = 0.5
        
        # Calculate confidence based on agreement
        if len(model_scores) > 0:
            scores_list = list(model_scores.values())
            if len(scores_list) > 1:
                std_dev = np.std(scores_list)
                # Lower std_dev = higher agreement = higher confidence
                confidence = max(0.3, 1.0 - std_dev * 2)  # Minimum 0.3
            else:
                # Single model - use score distance from 0.5 as confidence
                confidence = abs(scores_list[0] - 0.5) * 2
                confidence = max(0.3, confidence)
        else:
            # No models - low confidence but allow if metadata suggests fake
            confidence = max(0.2, metadata_score + camera_pipeline_score)
        
        # Boost confidence if ensemble_score is high
        if ensemble_score >= 0.5:
            confidence = max(confidence, 0.35)
        
        return min(max(ensemble_score, 0.0), 1.0), min(max(confidence, 0.0), 1.0)
    
    def _majority_vote(self,
                      model_scores: Dict[str, float],
                      metadata_score: float,
                      camera_pipeline_score: float) -> Tuple[float, float]:
        """Majority vote combination."""
        # Count votes
        fake_votes = 0
        real_votes = 0
        total_votes = 0
        
        # ML models vote
        for model_type, score in model_scores.items():
            if score > 0.5:
                fake_votes += 1
            else:
                real_votes += 1
            total_votes += 1
        
        # Metadata votes
        if metadata_score > 0.5:
            fake_votes += 1
        else:
            real_votes += 1
        total_votes += 1
        
        # Camera pipeline votes
        if camera_pipeline_score > 0.5:
            fake_votes += 1
        else:
            real_votes += 1
        total_votes += 1
        
        # Majority decides
        if fake_votes > real_votes:
            ensemble_score = 0.5 + (fake_votes / total_votes) * 0.5
        else:
            ensemble_score = 0.5 - (real_votes / total_votes) * 0.5
        
        # Confidence based on vote margin
        vote_margin = abs(fake_votes - real_votes) / total_votes
        confidence = vote_margin
        
        return ensemble_score, confidence
    
    def _max_vote(self,
                 model_scores: Dict[str, float],
                 metadata_score: float,
                 camera_pipeline_score: float) -> Tuple[float, float]:
        """Maximum score (any model says fake = fake)."""
        all_scores = list(model_scores.values()) + [metadata_score, camera_pipeline_score]
        if all_scores:
            ensemble_score = max(all_scores)
            confidence = ensemble_score  # Higher score = higher confidence
        else:
            ensemble_score = 0.5
            confidence = 0.5
        
        return ensemble_score, confidence
    
    def _consensus(self,
                  model_scores: Dict[str, float],
                  metadata_score: float,
                  camera_pipeline_score: float) -> Tuple[float, float]:
        """Consensus (all must agree)."""
        all_scores = list(model_scores.values()) + [metadata_score, camera_pipeline_score]
        
        if not all_scores:
            return 0.5, 0.5
        
        # Check if all agree (all > 0.5 or all < 0.5)
        all_fake = all(s > 0.5 for s in all_scores)
        all_real = all(s < 0.5 for s in all_scores)
        
        if all_fake:
            ensemble_score = np.mean(all_scores)
            confidence = 1.0
        elif all_real:
            ensemble_score = np.mean(all_scores)
            confidence = 1.0
        else:
            # No consensus - use average but low confidence
            ensemble_score = np.mean(all_scores)
            confidence = 0.3
        
        return ensemble_score, confidence

