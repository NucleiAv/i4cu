"""
Image Deepfake Detector
Analyzes images using EXIF data, OCR, and ML models.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import pytesseract
import numpy as np
from pathlib import Path


class ImageDetector:
    """Detects deepfakes in images using multiple techniques."""
    
    def __init__(self, ml_model=None, threshold: float = 0.52, confidence_threshold: float = 0.5):
        """
        Initialize the image detector.
        
        Args:
            ml_model: Optional pre-trained ML model for deepfake detection
            threshold: Detection threshold (default: 0.52, conservative to reduce false positives)
            confidence_threshold: Minimum confidence required (default: 0.5, higher to reduce false positives)
        """
        self.ml_model = ml_model
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        self.suspicious_exif_tags = [
            'Software', 'Artist', 'Copyright', 'ImageDescription',
            'Make', 'Model', 'DateTime', 'DateTimeOriginal', 'DateTimeDigitized'
        ]
    
    def detect(self, image_path: str) -> Dict:
        """
        Perform comprehensive deepfake detection on an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing detection results and scores
        """
        results = {
            'file_path': image_path,
            'file_type': 'image',
            'exif_analysis': {},
            'ocr_analysis': {},
            'ml_prediction': {},
            'overall_score': 0.0,
            'is_deepfake': False,
            'confidence': 0.0,
            'warnings': []
        }
        
        try:
            # Load image
            img = Image.open(image_path)
            
            # EXIF analysis
            exif_results = self._analyze_exif(img)
            results['exif_analysis'] = exif_results
            
            # OCR analysis
            ocr_results = self._analyze_ocr(img)
            results['ocr_analysis'] = ocr_results
            
            # ML model prediction
            if self.ml_model:
                ml_results = self._ml_predict(img)
                results['ml_prediction'] = ml_results
            else:
                # Use auto model loading (tries real models first, falls back to heuristic)
                from .model_loader import load_default_image_model
                auto_model = load_default_image_model('auto')
                self.ml_model = auto_model
                ml_results = self._ml_predict(img)
                results['ml_prediction'] = ml_results
            
            # Calculate overall score
            results['overall_score'], results['confidence'] = self._calculate_score(
                exif_results, ocr_results, results['ml_prediction']
            )
            
            # Determine if deepfake with confidence-based thresholding
            # Use instance thresholds (can be customized)
            
            # Flag as deepfake with conservative approach to minimize false positives
            # Always require both score AND confidence to be high
            score_above_threshold = results['overall_score'] > self.threshold
            
            if score_above_threshold:
                # Require high confidence to reduce false positives
                # Only be lenient for very high scores
                if results['overall_score'] > self.threshold + 0.12:
                    # Very high score (>0.64) - can be slightly lenient
                    results['is_deepfake'] = results['confidence'] > (self.confidence_threshold * 0.8)
                elif results['overall_score'] > self.threshold + 0.08:
                    # High score (>0.60) - require high confidence
                    results['is_deepfake'] = results['confidence'] > (self.confidence_threshold * 0.9)
                else:
                    # Score just above threshold - require full confidence to minimize false positives
                    results['is_deepfake'] = results['confidence'] > self.confidence_threshold
            else:
                results['is_deepfake'] = False
            
            # Collect warnings
            results['warnings'] = self._collect_warnings(exif_results, ocr_results)
            
        except Exception as e:
            results['error'] = str(e)
            results['warnings'].append(f"Error processing image: {str(e)}")
        
        return results
    
    def _analyze_exif(self, img: Image.Image) -> Dict:
        """Analyze EXIF data for suspicious patterns."""
        exif_data = {}
        suspicious_score = 0.0
        findings = []
        
        try:
            exif = img._getexif()
            
            if exif is not None:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    if tag in self.suspicious_exif_tags:
                        exif_data[tag] = str(value)
                        
                        # Check for suspicious patterns
                        if tag == 'Software':
                            software_lower = str(value).lower()
                            suspicious_keywords = ['deepfake', 'face swap', 'faceswap', 
                                                 'deepfacelab', 'fakeapp', 'refacer']
                            if any(keyword in software_lower for keyword in suspicious_keywords):
                                suspicious_score += 0.3
                                findings.append(f"Suspicious software detected: {value}")
                        
                        if tag == 'Artist' or tag == 'Copyright':
                            if 'ai' in str(value).lower() or 'generated' in str(value).lower():
                                suspicious_score += 0.2
                                findings.append(f"Suspicious metadata in {tag}: {value}")
                
                # Check for missing or inconsistent EXIF data
                if 'DateTime' not in exif_data and 'DateTimeOriginal' not in exif_data:
                    suspicious_score += 0.1
                    findings.append("Missing timestamp information")
                
                # Check for unusual camera models
                if 'Make' in exif_data or 'Model' in exif_data:
                    make_model = f"{exif_data.get('Make', '')} {exif_data.get('Model', '')}".lower()
                    if 'unknown' in make_model or 'virtual' in make_model:
                        suspicious_score += 0.2
                        findings.append("Unusual camera model detected")
            else:
                suspicious_score += 0.15
                findings.append("No EXIF data found (may indicate manipulation)")
        
        except Exception as e:
            findings.append(f"Error reading EXIF: {str(e)}")
        
        return {
            'exif_data': exif_data,
            'suspicious_score': min(suspicious_score, 1.0),
            'findings': findings
        }
    
    def _analyze_ocr(self, img: Image.Image) -> Dict:
        """Perform OCR analysis to detect text artifacts."""
        ocr_results = {
            'text_found': False,
            'text_content': '',
            'suspicious_score': 0.0,
            'findings': []
        }
        
        try:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Perform OCR
            text = pytesseract.image_to_string(img)
            text = text.strip()
            
            if text:
                ocr_results['text_found'] = True
                ocr_results['text_content'] = text[:200]  # Limit length
                
                # Check for suspicious text patterns
                text_lower = text.lower()
                suspicious_patterns = [
                    'deepfake', 'generated', 'ai created', 'synthetic',
                    'fake', 'manipulated', 'edited'
                ]
                
                for pattern in suspicious_patterns:
                    if pattern in text_lower:
                        ocr_results['suspicious_score'] += 0.2
                        ocr_results['findings'].append(f"Suspicious text pattern found: {pattern}")
                
                # Check for watermark-like text
                if 'watermark' in text_lower or 'copyright' in text_lower:
                    ocr_results['suspicious_score'] += 0.1
                    ocr_results['findings'].append("Watermark or copyright text detected")
            
            # Check for unusual text positioning (might indicate artifacts)
            # This is a simplified check - in production, you'd analyze text bounding boxes
            if len(text) > 100:  # Unusually long text might indicate artifacts
                ocr_results['suspicious_score'] += 0.1
                ocr_results['findings'].append("Unusually long text detected")
        
        except Exception as e:
            ocr_results['findings'].append(f"OCR error: {str(e)}")
        
        ocr_results['suspicious_score'] = min(ocr_results['suspicious_score'], 1.0)
        return ocr_results
    
    def _ml_predict(self, img: Image.Image) -> Dict:
        """Use ML model to predict if image is a deepfake."""
        if self.ml_model is None:
            return {'score': 0.5, 'confidence': 0.0, 'note': 'No ML model available'}
        
        try:
            # Preprocess image for model
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized)
            
            # Normalize to 0-1 if needed
            if img_array.max() > 1.0:
                img_array = img_array.astype(np.float32) / 255.0
            
            # Handle different model types
            if hasattr(self.ml_model, 'predict'):
                # Standard predict interface
                prediction = self.ml_model.predict(img_array)
            elif callable(self.ml_model):
                # Model is callable
                prediction = self.ml_model(img_array)
            else:
                # Try to use as-is
                prediction = self.ml_model
            
            # Extract score from prediction
            if isinstance(prediction, np.ndarray):
                if prediction.ndim > 1:
                    score = float(prediction[0][0] if prediction.shape[1] > 1 else prediction[0])
                else:
                    score = float(prediction[0] if len(prediction) > 0 else prediction)
            elif isinstance(prediction, (list, tuple)):
                score = float(prediction[0] if len(prediction) > 0 else 0.5)
            else:
                score = float(prediction)
            
            # Ensure score is in [0, 1] range
            score = max(0.0, min(1.0, score))
            
            # Calculate confidence
            confidence = abs(score - 0.5) * 2  # Higher confidence when score is far from 0.5
            
            return {
                'score': score,
                'confidence': confidence,
                'prediction': 'deepfake' if score > 0.5 else 'real',
                'model_type': getattr(self.ml_model, 'name', 'unknown')
            }
        
        except Exception as e:
            # On error, return moderate suspicion
            return {
                'score': 0.4,
                'confidence': 0.2,
                'error': str(e),
                'prediction': 'unknown'
            }
    
    def _calculate_score(self, exif_results: Dict, ocr_results: Dict, ml_results: Dict) -> Tuple[float, float]:
        """
        Calculate overall deepfake score from all analyses.
        
        Returns:
            Tuple of (overall_score, confidence)
        """
        # Adjusted weights - trust ML model heavily, minimal EXIF/OCR influence
        # Optimized for high recall (catching deepfakes)
        weights = {
            'exif': 0.05,  # Minimal - EXIF can be unreliable
            'ocr': 0.05,   # Minimal - OCR can give false positives
            'ml': 0.9      # Maximum trust in ML model - it's the most reliable
        }
        
        exif_score = exif_results.get('suspicious_score', 0.0)
        ocr_score = ocr_results.get('suspicious_score', 0.0)
        ml_score = ml_results.get('score', 0.5)
        
        # For CLIP-ViT and other modern models, the score interpretation is different
        # CLIP-ViT gives similarity scores that need calibration
        model_type = ml_results.get('model_type', 'unknown')
        
        # Use CLIP-ViT scores with conservative calibration to reduce false positives
        # CLIP can give false positives, so we need to be more careful
        if 'CLIP' in model_type or 'clip' in model_type.lower():
            # CLIP scores - conservative approach to minimize false positives
            if ml_score > 0.6:
                # For high scores (>0.6), use them more directly (likely real deepfakes)
                ml_contribution = 0.50 + (ml_score - 0.6) * 1.25  # Scale 0.6-1.0 to 0.50-1.0
            elif ml_score > 0.5:
                # For moderate scores (0.5-0.6), be conservative
                ml_contribution = 0.40 + (ml_score - 0.5) * 1.0  # Scale 0.5-0.6 to 0.40-0.50
            else:
                # For scores below 0.5, they're likely real - scale down significantly
                ml_contribution = ml_score * 0.7  # Strong scaling down to reduce false positives
        else:
            # For other models, use score directly but be more conservative
            if ml_score == 0.5:
                ml_contribution = 0.3  # Reduced from 0.4 - less suspicion when uncertain
            elif ml_score > 0.5:
                ml_contribution = ml_score
            else:
                # For low scores (likely real), scale down to avoid false positives
                ml_contribution = ml_score * 0.7
        
        # Combine scores
        overall_score = (
            exif_score * weights['exif'] +
            ocr_score * weights['ocr'] +
            ml_contribution * weights['ml']
        )
        
        # Boost score conservatively - only when very confident
        # Prioritize reducing false positives over catching all deepfakes
        ml_confident = ml_contribution > 0.6  # Higher threshold - only boost when very confident
        indicator_count = sum([
            exif_score > 0.3,  # Higher threshold
            ocr_score > 0.3,   # Higher threshold
            ml_confident
        ])
        
        # Only boost if ML is very confident AND multiple indicators agree
        # This prevents false positives from weak signals
        if ml_confident and indicator_count >= 2:
            overall_score = min(overall_score * 1.08, 1.0)  # Conservative boost
        
        # No additional boost for borderline cases - too risky for false positives
        
        # Calculate confidence - emphasize ML confidence more
        ml_confidence = ml_results.get('confidence', 0.5)
        signal_strength = max(exif_score, ocr_score, ml_contribution)
        
        # Weight confidence more heavily on ML model confidence
        confidence = (signal_strength * 0.4 + ml_confidence * 0.6)
        confidence = min(confidence, 1.0)
        
        return min(overall_score, 1.0), confidence
    
    def _collect_warnings(self, exif_results: Dict, ocr_results: Dict) -> List[str]:
        """Collect all warnings from different analyses."""
        warnings = []
        
        warnings.extend(exif_results.get('findings', []))
        warnings.extend(ocr_results.get('findings', []))
        
        return warnings

