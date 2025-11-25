"""
Video Deepfake Detector
Analyzes videos by extracting frames and applying image detection techniques.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import tempfile
from .image_detector import ImageDetector


class VideoDetector:
    """Detects deepfakes in videos by analyzing extracted frames."""
    
    def __init__(self, image_detector: Optional[ImageDetector] = None, frames_per_second: float = 1.0):
        """
        Initialize the video detector.
        
        Args:
            image_detector: ImageDetector instance for frame analysis
            frames_per_second: How many frames per second to extract for analysis
        """
        self.image_detector = image_detector or ImageDetector()
        self.frames_per_second = frames_per_second
    
    def detect(self, video_path: str, max_frames: int = 30) -> Dict:
        """
        Perform comprehensive deepfake detection on a video.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Dictionary containing detection results and scores
        """
        results = {
            'file_path': video_path,
            'file_type': 'video',
            'video_info': {},
            'frame_analyses': [],
            'overall_score': 0.0,
            'is_deepfake': False,
            'confidence': 0.0,
            'warnings': []
        }
        
        try:
            # Get video information
            video_info = self._get_video_info(video_path)
            results['video_info'] = video_info
            
            # Extract frames
            frames = self._extract_frames(video_path, max_frames)
            
            if not frames:
                results['warnings'].append("Could not extract frames from video")
                return results
            
            # Analyze each frame
            frame_scores = []
            for i, frame_path in enumerate(frames):
                try:
                    frame_result = self.image_detector.detect(frame_path)
                    frame_result['frame_number'] = i
                    results['frame_analyses'].append(frame_result)
                    frame_scores.append(frame_result.get('overall_score', 0.0))
                except Exception as e:
                    results['warnings'].append(f"Error analyzing frame {i}: {str(e)}")
            
            # Calculate overall video score
            if frame_scores:
                results['overall_score'] = np.mean(frame_scores)
                # Use max score as well (if any frame is suspicious, video is suspicious)
                max_score = np.max(frame_scores)
                # Weighted average: 70% mean, 30% max (to catch videos with occasional deepfake frames)
                results['overall_score'] = results['overall_score'] * 0.7 + max_score * 0.3
                results['confidence'] = 1.0 - min(np.std(frame_scores), 0.5)  # Higher confidence if consistent
                # Use same threshold as image detector, but be more conservative for videos
                # Videos need higher thresholds to reduce false positives
                image_threshold = getattr(self.image_detector, 'threshold', 0.55)
                image_conf_threshold = getattr(self.image_detector, 'confidence_threshold', 0.6)
                
                # For videos, require higher scores to reduce false positives
                # Also require that a significant portion of frames are flagged
                video_threshold = max(image_threshold, 0.55)  # Minimum 0.55 for videos
                video_conf_threshold = max(image_conf_threshold, 0.60)  # Minimum 0.60 for videos
                
                # Count how many frames are above threshold
                flagged_frames = sum(1 for s in frame_scores if s > video_threshold)
                frame_ratio = flagged_frames / len(frame_scores) if frame_scores else 0
                
                # Require both high score AND significant portion of frames flagged
                results['is_deepfake'] = (
                    results['overall_score'] >= video_threshold and
                    results['confidence'] >= video_conf_threshold and
                    frame_ratio >= 0.3  # At least 30% of frames must be flagged
                )
                
                # Add statistics
                results['statistics'] = {
                    'frames_analyzed': len(frame_scores),
                    'mean_score': float(np.mean(frame_scores)),
                    'std_score': float(np.std(frame_scores)),
                    'min_score': float(np.min(frame_scores)),
                    'max_score': float(np.max(frame_scores)),
                    'deepfake_frames': sum(1 for s in frame_scores if s > 0.6)
                }
            
            # Clean up temporary frame files
            for frame_path in frames:
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                except:
                    pass
        
        except Exception as e:
            results['error'] = str(e)
            results['warnings'].append(f"Error processing video: {str(e)}")
        
        return results
    
    def _get_video_info(self, video_path: str) -> Dict:
        """Extract video metadata and information."""
        info = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info['duration_seconds'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
                
                # Get codec information
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                info['codec'] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
        
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def _extract_frames(self, video_path: str, max_frames: int) -> List[str]:
        """Extract frames from video at specified intervals."""
        frames = []
        temp_dir = tempfile.gettempdir()
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0 or total_frames == 0:
                cap.release()
                return frames
            
            # Calculate frame interval
            frame_interval = max(1, int(fps / self.frames_per_second))
            
            # Limit total frames
            frames_to_extract = min(max_frames, total_frames // frame_interval)
            if frames_to_extract == 0:
                frames_to_extract = min(max_frames, total_frames)
                frame_interval = max(1, total_frames // frames_to_extract)
            
            frame_count = 0
            extracted_count = 0
            
            while extracted_count < frames_to_extract:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Save frame to temporary file
                    frame_path = os.path.join(
                        temp_dir,
                        f"frame_{extracted_count}_{os.path.basename(video_path)}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                    frames.append(frame_path)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
        
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
        
        return frames

