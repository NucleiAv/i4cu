"""
Audio Ensemble Deepfake Detector
Two-stage audio pipeline:
- Stage 1: Fast AASIST (MTUCI/AASIST3 on Hugging Face)
- Stage 2: High-confidence Wav2Vec2/WavLM-style model for deepfake audio classification
"""

import os
from typing import Dict, Optional

import numpy as np
import librosa
import soundfile as sf

from .model_loader import ModelLoader


class AudioEnsembleDetector:
    """
    Ensemble detector for audio deepfakes.
    
    Stage 1: AASIST (fast, good recall)
    Stage 2: Wav2Vec2/WavLM-style classifier (higher confidence)
    
    Final decision is based on a weighted combination of both.
    """

    def __init__(self):
        self.loader = ModelLoader()

        # Stage 1: AASIST3 from Hugging Face (MTUCI/AASIST3)
        # This is treated as the fast, primary detector.
        self.aasist_model = self.loader.load_audio_model("aasist")

        # Stage 2: High-confidence model.
        # Use a Wav2Vec2-based deepfake classifier as a proxy for WavLM/HuBERT:
        #   Gustking/wav2vec2-large-xlsr-deepfake-audio-classification
        # You can swap this to another HF repo if you prefer.
        self.high_conf_model = self.loader.load_audio_model("wav2vec2_deepfake")

    def detect(self, audio_path: str) -> Dict:
        """
        Perform ensemble deepfake detection on an audio file.

        Returns a result dictionary compatible with the existing AudioDetector output.
        """
        results: Dict = {
            "file_path": audio_path,
            "file_type": "audio",
            "audio_info": {},
            "ml_prediction": {},
            "overall_score": 0.0,
            "is_deepfake": False,
            "confidence": 0.0,
            "warnings": [],
        }

        try:
            if not os.path.exists(audio_path):
                results["warnings"].append("File does not exist")
                return results

            # Load waveform (keep original sample rate for info)
            y, sr = librosa.load(audio_path, sr=None, mono=True)

            # Basic audio info
            results["audio_info"] = self._get_audio_info(audio_path, y, sr)

            # For the models, resample to 16 kHz which most HF audio models expect
            target_sr = 16000
            if sr != target_sr:
                y_model = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                model_sr = target_sr
            else:
                y_model = y
                model_sr = sr

            model_scores: Dict[str, float] = {}
            model_confidences: Dict[str, float] = {}

            # ----- Stage 1: AASIST (fast pass) -----
            aasist_score = 0.5
            aasist_conf = 0.0
            if self.aasist_model is not None:
                try:
                    aasist_score = float(
                        self.aasist_model.predict_waveform(y_model, model_sr)
                    )
                    aasist_score = max(0.0, min(1.0, aasist_score))
                    # Confidence: distance from 0.5
                    aasist_conf = max(0.3, abs(aasist_score - 0.5) * 2.0)
                except Exception as e:  # pragma: no cover - defensive
                    results["warnings"].append(f"AASIST inference error: {e}")
            else:
                results["warnings"].append("AASIST model not loaded")

            model_scores["aasist"] = aasist_score
            model_confidences["aasist"] = aasist_conf

            # If AASIST is very confident, short-circuit
            if aasist_score >= 0.80 and aasist_conf >= 0.60:
                ensemble_score = aasist_score
                confidence = aasist_conf
                is_deepfake = True
            else:
                # ----- Stage 2: High-confidence model (Wav2Vec2 / WavLM-style) -----
                high_score = aasist_score
                high_conf = aasist_conf

                if self.high_conf_model is not None:
                    try:
                        hs = float(
                            self.high_conf_model.predict_waveform(y_model, model_sr)
                        )
                        hs = max(0.0, min(1.0, hs))
                        hc = max(0.3, abs(hs - 0.5) * 2.0)

                        model_scores["wav2vec2_deepfake"] = hs
                        model_confidences["wav2vec2_deepfake"] = hc

                        high_score = hs
                        high_conf = hc
                    except Exception as e:  # pragma: no cover - defensive
                        results["warnings"].append(
                            f"High-confidence audio model error: {e}"
                        )
                else:
                    results["warnings"].append("High-confidence audio model not loaded")

                # Combine stage 1 and stage 2
                if "wav2vec2_deepfake" in model_scores:
                    ensemble_score = (
                        0.6 * aasist_score + 0.4 * model_scores["wav2vec2_deepfake"]
                    )
                    # Confidence based on agreement and individual confidences
                    agreement = 1.0 - abs(aasist_score - model_scores["wav2vec2_deepfake"])
                    confidence = min(
                        1.0,
                        0.5 * aasist_conf + 0.5 * high_conf + 0.3 * agreement,
                    )
                else:
                    ensemble_score = aasist_score
                    confidence = aasist_conf

                # Decision logic (conservative to avoid too many false positives)
                strong_indicators = [
                    ensemble_score >= 0.75 and confidence >= 0.55,
                    ensemble_score >= 0.70
                    and confidence >= 0.50
                    and (aasist_score >= 0.70 or high_score >= 0.70),
                ]

                is_deepfake = any(strong_indicators)

            # Populate result
            results["overall_score"] = float(ensemble_score)
            results["confidence"] = float(confidence)
            results["is_deepfake"] = bool(is_deepfake)

            results["ml_prediction"] = {
                "score": float(ensemble_score),
                "confidence": float(confidence),
                "prediction": "deepfake" if is_deepfake else "real",
                "model_scores": model_scores,
                "model_confidences": model_confidences,
            }

        except Exception as e:  # pragma: no cover - defensive
            results["error"] = str(e)
            results["warnings"].append(f"Error processing audio: {e}")

        return results

    def _get_audio_info(self, audio_path: str, y: np.ndarray, sr: int) -> Dict:
        """Extract audio file metadata and information (shared with simple AudioDetector style)."""
        info = {
            "sample_rate": int(sr),
            "duration_seconds": float(len(y) / sr) if sr > 0 else 0.0,
            "channels": 1 if y.ndim == 1 else int(y.shape[0]),
            "file_size_bytes": os.path.getsize(audio_path)
            if os.path.exists(audio_path)
            else 0,
        }

        try:
            # Try to get metadata using soundfile
            with sf.SoundFile(audio_path) as f:
                info["subtype"] = f.subtype
                info["format"] = f.format
                info["sections"] = f.sections
        except Exception:
            # Non-fatal
            pass

        return info


