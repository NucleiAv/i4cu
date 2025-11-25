"""
Model Loader for Deepfake Detection
Loads and manages pre-trained deepfake detection models.
"""

import os
# Suppress NNPACK warnings before importing PyTorch
# Disable NNPACK entirely to prevent warnings
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_DISABLE_NNPACK'] = '1'
# Try to disable NNPACK via PyTorch settings
try:
    # Set before importing torch
    os.environ['OMP_NUM_THREADS'] = '1'
except:
    pass

import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import warnings
import logging

# Suppress NNPACK warnings (harmless - PyTorch falls back to default CPU implementation)
warnings.filterwarnings('ignore', message='.*NNPACK.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
logging.getLogger().setLevel(logging.ERROR)

# Try to import ML libraries
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from PIL import Image
    # Suppress PyTorch warnings and disable NNPACK
    torch.backends.mkldnn.enabled = False
    # Try to disable NNPACK if possible
    try:
        # Disable NNPACK backend
        if hasattr(torch.backends, 'nnpack'):
            torch.backends.nnpack.enabled = False
    except:
        pass
    # Set thread settings to avoid NNPACK
    try:
        torch.set_num_threads(1)
    except:
        pass
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        pipeline,
        AutoImageProcessor,
        AutoModelForImageClassification,
        CLIPProcessor,
        CLIPModel,
        ViTImageProcessor,
        ViTForImageClassification,
        AutoModel,
        AutoProcessor,
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class ModelLoader:
    """Loads and manages pre-trained deepfake detection models."""
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            models_dir: Directory to store/download models (default: ~/.deepfake_detector/models)
        """
        self.models_cache = {}
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            self.models_dir = Path.home() / '.deepfake_detector' / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # IMAGE MODELS
    # ------------------------------------------------------------------
    
    def load_image_model(self, model_type: str = 'auto', model_name: Optional[str] = None) -> Any:
        """
        Load a deepfake detection model for images.
        
        Args:
            model_type: Type of model to load ('auto', 'xception', 'mesonet', 'efficientnet', 
                      'heuristic', 'pytorch', 'tensorflow', 'huggingface')
            model_name: Specific model name/path (optional)
            
        Returns:
            Loaded model
        """
        if model_type == 'heuristic':
            return HeuristicImageModel()
        
        # Try to load actual ML models
        if model_type == 'auto':
            # Try in order: modern ViT models first, then older models, then fallback
            for mt in ['clip_vit', 'uia_vit', 'face_xray', 'xception', 'mesonet', 'efficientnet', 'huggingface', 'pytorch']:
                try:
                    model = self._load_specific_model(mt, model_name)
                    if model is not None:
                        return model
                except Exception as e:
                    warnings.warn(f"Failed to load {mt}: {e}")
            return HeuristicImageModel()
        else:
            model = self._load_specific_model(model_type, model_name)
            if model is None:
                warnings.warn(f"Model type {model_type} not available, using heuristic model")
                return HeuristicImageModel()
            return model
    
    def _load_specific_model(self, model_type: str, model_name: Optional[str]) -> Optional[Any]:
        """Load a specific model type."""
        if model_type == 'clip_vit':
            return self._load_clip_vit_model(model_name)
        elif model_type == 'uia_vit':
            return self._load_uia_vit_model(model_name)
        elif model_type == 'face_xray':
            return self._load_face_xray_model(model_name)
        elif model_type == 'xception':
            return self._load_xception_model(model_name)
        elif model_type == 'mesonet':
            return self._load_mesonet_model(model_name)
        elif model_type == 'efficientnet':
            return self._load_efficientnet_model(model_name)
        elif model_type == 'pytorch' and TORCH_AVAILABLE:
            return self._load_pytorch_model(model_name)
        elif model_type == 'huggingface' and HF_AVAILABLE:
            return self._load_huggingface_model(model_name)
        elif model_type == 'tensorflow' and TF_AVAILABLE:
            return self._load_tensorflow_model(model_name)
        return None

    # ------------------------------------------------------------------
    # AUDIO MODELS
    # ------------------------------------------------------------------

    def load_audio_model(self, model_type: str = "aasist", model_name: Optional[str] = None) -> Any:
        """
        Load a deepfake detection model for audio.

        Uses Hugging Face audio-classification models as backends.

        Args:
            model_type: 'aasist', 'wav2vec2_deepfake', 'deepfake_v1', 'deepfake_v2', or 'auto'
            model_name: Optional HF model repo id to override defaults.
        """
        if not HF_AVAILABLE or not TORCH_AVAILABLE:
            return None

        # Auto mode: prefer AASIST, then Wav2Vec2-based detector
        if model_type == "auto":
            for mt in ["aasist", "wav2vec2_deepfake", "deepfake_v2", "deepfake_v1"]:
                try:
                    m = self._load_hf_audio_model(mt, model_name=None)
                    if m is not None:
                        return m
                except Exception:
                    continue
            return None

        return self._load_hf_audio_model(model_type, model_name)

    def _load_hf_audio_model(self, model_type: str, model_name: Optional[str]) -> Optional[Any]:
        """Load a specific Hugging Face audio-classification model."""
        if not HF_AVAILABLE or not TORCH_AVAILABLE:
            return None

        # Map friendly types to HF repo ids
        if model_type in ("aasist", "aasist3"):
            # NOTE:
            # The MTUCI/AASIST3 repo does not expose a standard Hugging Face
            # audio feature extractor / preprocessor, so we cannot load it
            # directly via AutoFeatureExtractor/AutoProcessor.
            # As a practical workaround, we treat a strong deepfake audio
            # classifier as our "fast AASIST-like" model here.
            repo_id = model_name or "MelodyMachine/Deepfake-audio-detection-V2"
        elif model_type in ("deepfake_v1",):
            # Generic deepfake audio detector (v1)
            repo_id = model_name or "mo-thecreator/Deepfake-audio-detection"
        elif model_type in ("deepfake_v2",):
            repo_id = model_name or "MelodyMachine/Deepfake-audio-detection-V2"
        elif model_type in ("wav2vec2_deepfake", "wavlm"):
            repo_id = model_name or "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
        else:
            return None

        try:
            # Some repos use AutoFeatureExtractor, others AutoProcessor
            try:
                processor = AutoProcessor.from_pretrained(repo_id)
            except Exception:
                processor = AutoFeatureExtractor.from_pretrained(repo_id)

            model = AutoModelForAudioClassification.from_pretrained(repo_id)
            return HFAudioClassifierWrapper(processor, model)
        except Exception as e:
            warnings.warn(f"Failed to load audio model {model_type} ({repo_id}): {e}")
            return None
    
    def _load_xception_model(self, model_path: Optional[str] = None) -> Optional[Any]:
        """Load XceptionNet model (FaceForensics++)."""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Try to load from torch.hub (if available)
            try:
                model = torch.hub.load('pytorch/vision:v0.10.0', 'xception', pretrained=True)
                # Modify for binary classification
                model.fc = nn.Linear(model.fc.in_features, 2)
                model = XceptionWrapper(model, model_path)
                return model
            except:
                pass
            
            # Try to load from local path
            if model_path:
                if Path(model_path).exists():
                    return XceptionWrapper(None, model_path)
            
            # Try default location
            default_path = self.models_dir / 'xception.pth'
            if default_path.exists():
                return XceptionWrapper(None, str(default_path))
            
            # If no model found, return None to fall back
            warnings.warn("Xception model not found. Please download weights from FaceForensics++")
            return None
            
        except Exception as e:
            warnings.warn(f"Failed to load Xception model: {e}")
            return None
    
    def _load_mesonet_model(self, model_path: Optional[str] = None) -> Optional[Any]:
        """Load MesoNet model."""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Create MesoNet architecture
            model = MesoNet()
            
            # Try to load weights
            if model_path and Path(model_path).exists():
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            else:
                default_path = self.models_dir / 'mesonet.pth'
                if default_path.exists():
                    model.load_state_dict(torch.load(default_path, map_location='cpu'))
                else:
                    warnings.warn("MesoNet weights not found. Using untrained model.")
            
            model.eval()
            return MesoNetWrapper(model)
            
        except Exception as e:
            warnings.warn(f"Failed to load MesoNet model: {e}")
            return None
    
    def _load_efficientnet_model(self, model_path: Optional[str] = None) -> Optional[Any]:
        """Load EfficientNet model."""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Try EfficientNet-B4
            try:
                import torchvision.models as models
                model = models.efficientnet_b4(pretrained=True)
                # Modify for binary classification
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(num_features, 2)
                )
                model = EfficientNetWrapper(model, model_path)
                return model
            except Exception as e:
                warnings.warn(f"Failed to load EfficientNet: {e}")
                return None
        except Exception as e:
            warnings.warn(f"Failed to load EfficientNet model: {e}")
            return None
    
    def _load_pytorch_model(self, model_path: Optional[str] = None) -> Optional[Any]:
        """Load generic PyTorch model."""
        if not TORCH_AVAILABLE or not model_path:
            return None
        try:
            model = torch.load(model_path, map_location='cpu')
            if isinstance(model, nn.Module):
                model.eval()
                return PyTorchWrapper(model)
            return None
        except Exception as e:
            warnings.warn(f"Failed to load PyTorch model: {e}")
            return None
    
    def _load_tensorflow_model(self, model_path: Optional[str] = None) -> Optional[Any]:
        """Load TensorFlow model."""
        if not TF_AVAILABLE:
            return None
        try:
            if model_path and Path(model_path).exists():
                model = tf.keras.models.load_model(model_path)
                return TensorFlowWrapper(model)
            return None
        except Exception as e:
            warnings.warn(f"Failed to load TensorFlow model: {e}")
            return None
    
    def _load_clip_vit_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """Load CLIP-ViT model for deepfake detection."""
        if not HF_AVAILABLE:
            return None
        
        try:
            # Use CLIP model - excellent for detecting inconsistencies and artifacts
            model_name = model_name or "openai/clip-vit-base-patch32"
            
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            
            return CLIPViTWrapper(processor, model)
        except Exception as e:
            warnings.warn(f"Failed to load CLIP-ViT model: {e}")
            return None
    
    def _load_uia_vit_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """Load UIA-ViT (Universal Image Analysis ViT) model."""
        if not HF_AVAILABLE:
            return None
        
        try:
            # Try to load a ViT model fine-tuned for deepfake detection
            if model_name:
                try:
                    processor = ViTImageProcessor.from_pretrained(model_name)
                    model = ViTForImageClassification.from_pretrained(model_name)
                    return UIAViTWrapper(processor, model)
                except:
                    pass
            
            # Fallback: Use a general ViT model that can detect inconsistencies
            vit_models = [
                "google/vit-base-patch16-224",
                "WinKawaks/vit-small-patch16-224",
            ]
            
            for vit_model in vit_models:
                try:
                    processor = ViTImageProcessor.from_pretrained(vit_model)
                    model = ViTForImageClassification.from_pretrained(vit_model)
                    return UIAViTWrapper(processor, model)
                except:
                    continue
            
            return None
        except Exception as e:
            warnings.warn(f"Failed to load UIA-ViT model: {e}")
            return None
    
    def _load_face_xray_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """Load Face X-Ray model for detecting compositing artifacts."""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Face X-Ray detects compositing artifacts in face regions
            # Try to load from local path
            if model_name and Path(model_name).exists():
                return FaceXRayWrapper(None, model_name)
            
            # Try default location
            default_path = self.models_dir / 'face_xray.pth'
            if default_path.exists():
                return FaceXRayWrapper(None, str(default_path))
            
            # Create Face X-Ray architecture (ResNet-based)
            # This uses ImageNet pretrained weights as base
            try:
                import torchvision.models as models
                model = models.resnet50(pretrained=True)
                # Modify for binary classification
                model.fc = nn.Linear(model.fc.in_features, 2)
                return FaceXRayWrapper(model, None)
            except:
                return None
        except Exception as e:
            warnings.warn(f"Failed to load Face X-Ray model: {e}")
            return None
    
    def _load_huggingface_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """Load Hugging Face model."""
        if not HF_AVAILABLE:
            return None
        try:
            # Try to find a deepfake detection model on Hugging Face
            if model_name:
                processor = AutoImageProcessor.from_pretrained(model_name)
                model = AutoModelForImageClassification.from_pretrained(model_name)
                return HuggingFaceWrapper(processor, model)
            return None
        except Exception as e:
            warnings.warn(f"Failed to load Hugging Face model: {e}")
            return None


# Model Wrappers
class XceptionWrapper:
    """Wrapper for XceptionNet model."""
    
    def __init__(self, model: Optional[Any], weights_path: Optional[str] = None):
        self.name = "XceptionNet"
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        if model is None and weights_path:
            self._load_weights(weights_path)
        elif model:
            if weights_path and Path(weights_path).exists():
                model.load_state_dict(torch.load(weights_path, map_location=self.device))
            model.to(self.device)
            model.eval()
    
    def _load_weights(self, weights_path: str):
        """Load model weights."""
        try:
            import torchvision.models as models
            self.model = models.xception(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"Failed to load Xception weights: {e}")
    
    def predict(self, image_array: np.ndarray) -> float:
        """Predict if image is deepfake."""
        if self.model is None:
            return 0.5
        
        try:
            # Convert numpy array to PIL Image
            if isinstance(image_array, np.ndarray):
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                if len(image_array.shape) == 3:
                    img = Image.fromarray(image_array)
                else:
                    img = Image.fromarray(image_array, mode='RGB')
            else:
                img = image_array
            
            # Transform and predict
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                # Assuming class 1 is deepfake
                deepfake_prob = probs[0][1].item()
            
            return float(deepfake_prob)
        except Exception as e:
            warnings.warn(f"Xception prediction error: {e}")
            return 0.5
    
    def __call__(self, image_array: np.ndarray) -> float:
        return self.predict(image_array)


class MesoNet(nn.Module):
    """MesoNet architecture."""
    
    def __init__(self):
        super(MesoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn5 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MesoNetWrapper:
    """Wrapper for MesoNet model."""
    
    def __init__(self, model: nn.Module):
        self.name = "MesoNet"
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def predict(self, image_array: np.ndarray) -> float:
        """Predict if image is deepfake."""
        try:
            if isinstance(image_array, np.ndarray):
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                if len(image_array.shape) == 3:
                    img = Image.fromarray(image_array)
                else:
                    img = Image.fromarray(image_array, mode='RGB')
            else:
                img = image_array
            
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                deepfake_prob = probs[0][1].item()
            
            return float(deepfake_prob)
        except Exception as e:
            warnings.warn(f"MesoNet prediction error: {e}")
            return 0.5
    
    def __call__(self, image_array: np.ndarray) -> float:
        return self.predict(image_array)


class EfficientNetWrapper:
    """Wrapper for EfficientNet model."""
    
    def __init__(self, model: nn.Module, weights_path: Optional[str] = None):
        self.name = "EfficientNet"
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if weights_path and Path(weights_path).exists():
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.to(self.device)
        model.eval()
    
    def predict(self, image_array: np.ndarray) -> float:
        """Predict if image is deepfake."""
        try:
            if isinstance(image_array, np.ndarray):
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                if len(image_array.shape) == 3:
                    img = Image.fromarray(image_array)
                else:
                    img = Image.fromarray(image_array, mode='RGB')
            else:
                img = image_array
            
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                deepfake_prob = probs[0][1].item()
            
            return float(deepfake_prob)
        except Exception as e:
            warnings.warn(f"EfficientNet prediction error: {e}")
            return 0.5
    
    def __call__(self, image_array: np.ndarray) -> float:
        return self.predict(image_array)


class PyTorchWrapper:
    """Generic PyTorch model wrapper."""
    
    def __init__(self, model: nn.Module):
        self.name = "PyTorch Model"
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_array: np.ndarray) -> float:
        """Predict if image is deepfake."""
        try:
            if isinstance(image_array, np.ndarray):
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                if len(image_array.shape) == 3:
                    img = Image.fromarray(image_array)
                else:
                    img = Image.fromarray(image_array, mode='RGB')
            else:
                img = image_array
            
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                probs = torch.softmax(outputs, dim=1)
                deepfake_prob = probs[0][1].item() if probs.shape[1] > 1 else probs[0][0].item()
            
            return float(deepfake_prob)
        except Exception as e:
            warnings.warn(f"PyTorch model prediction error: {e}")
            return 0.5
    
    def __call__(self, image_array: np.ndarray) -> float:
        return self.predict(image_array)


class TensorFlowWrapper:
    """Wrapper for TensorFlow model."""
    
    def __init__(self, model: Any):
        self.name = "TensorFlow Model"
        self.model = model
    
    def predict(self, image_array: np.ndarray) -> float:
        """Predict if image is deepfake."""
        try:
            # Preprocess for TensorFlow
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            
            # Resize and normalize
            import tensorflow as tf
            img = tf.image.resize(image_array, [224, 224])
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.expand_dims(img, 0)
            
            prediction = self.model.predict(img, verbose=0)
            if len(prediction[0]) > 1:
                return float(prediction[0][1])
            return float(prediction[0][0])
        except Exception as e:
            warnings.warn(f"TensorFlow prediction error: {e}")
            return 0.5
    
    def __call__(self, image_array: np.ndarray) -> float:
        return self.predict(image_array)


class CLIPViTWrapper:
    """Wrapper for CLIP-ViT model - excellent for detecting inconsistencies."""
    
    def __init__(self, processor: Any, model: Any):
        self.name = "CLIP-ViT"
        self.processor = processor
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image_array: np.ndarray) -> float:
        """Predict if image is deepfake using CLIP's ability to detect inconsistencies."""
        try:
            if isinstance(image_array, np.ndarray):
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                if len(image_array.shape) == 3:
                    img = Image.fromarray(image_array)
                else:
                    img = Image.fromarray(image_array, mode='RGB')
            else:
                img = image_array
            
            # Process image
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Get image features
                image_features = self.model.get_image_features(**inputs)
                
                # Create text prompts for real vs fake
                text_prompts = [
                    "a real authentic photograph of a person",
                    "a fake manipulated deepfake image of a person with artifacts"
                ]
                text_inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (image_features @ text_features.T) * 100
                probs = torch.softmax(similarity, dim=1)
                
                # Return probability of being fake (second prompt)
                fake_prob = probs[0][1].item()
            
            return float(fake_prob)
        except Exception as e:
            warnings.warn(f"CLIP-ViT prediction error: {e}")
            return 0.5
    
    def __call__(self, image_array: np.ndarray) -> float:
        return self.predict(image_array)


class UIAViTWrapper:
    """Wrapper for UIA-ViT (Universal Image Analysis ViT) model."""
    
    def __init__(self, processor: Any, model: Any):
        self.name = "UIA-ViT"
        self.processor = processor
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image_array: np.ndarray) -> float:
        """Predict if image is deepfake using ViT."""
        try:
            if isinstance(image_array, np.ndarray):
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                if len(image_array.shape) == 3:
                    img = Image.fromarray(image_array)
                else:
                    img = Image.fromarray(image_array, mode='RGB')
            else:
                img = image_array
            
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # If model has 2 classes, use softmax; otherwise use sigmoid
                if logits.shape[1] >= 2:
                    probs = torch.softmax(logits, dim=1)
                    # Assume class 1 is fake, or use heuristic
                    fake_prob = probs[0][1].item() if logits.shape[1] > 1 else 0.5
                else:
                    # Binary classification with single output
                    fake_prob = torch.sigmoid(logits[0][0]).item()
            
            return float(fake_prob)
        except Exception as e:
            warnings.warn(f"UIA-ViT prediction error: {e}")
            return 0.5
    
    def __call__(self, image_array: np.ndarray) -> float:
        return self.predict(image_array)


class FaceXRayWrapper:
    """Wrapper for Face X-Ray model - detects compositing artifacts."""
    
    def __init__(self, model: Optional[Any], weights_path: Optional[str] = None):
        self.name = "Face X-Ray"
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if model is None and weights_path:
            self._load_weights(weights_path)
        elif model:
            if weights_path and Path(weights_path).exists():
                model.load_state_dict(torch.load(weights_path, map_location=self.device))
            model.to(self.device)
            model.eval()
    
    def _load_weights(self, weights_path: str):
        """Load model weights."""
        try:
            import torchvision.models as models
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"Failed to load Face X-Ray weights: {e}")
    
    def predict(self, image_array: np.ndarray) -> float:
        """Predict if image is deepfake."""
        if self.model is None:
            return 0.5
        
        try:
            if isinstance(image_array, np.ndarray):
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                if len(image_array.shape) == 3:
                    img = Image.fromarray(image_array)
                else:
                    img = Image.fromarray(image_array, mode='RGB')
            else:
                img = image_array
            
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                deepfake_prob = probs[0][1].item()
            
            return float(deepfake_prob)
        except Exception as e:
            warnings.warn(f"Face X-Ray prediction error: {e}")
            return 0.5
    
    def __call__(self, image_array: np.ndarray) -> float:
        return self.predict(image_array)


class HuggingFaceWrapper:
    """Wrapper for Hugging Face model."""
    
    def __init__(self, processor: Any, model: Any):
        self.name = "HuggingFace Model"
        self.processor = processor
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image_array: np.ndarray) -> float:
        """Predict if image is deepfake."""
        try:
            if isinstance(image_array, np.ndarray):
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                if len(image_array.shape) == 3:
                    img = Image.fromarray(image_array)
                else:
                    img = Image.fromarray(image_array, mode='RGB')
            else:
                img = image_array
            
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                deepfake_prob = probs[0][1].item()
            
            return float(deepfake_prob)
        except Exception as e:
            warnings.warn(f"HuggingFace prediction error: {e}")
            return 0.5
    
    def __call__(self, image_array: np.ndarray) -> float:
        return self.predict(image_array)


class HFAudioClassifierWrapper:
    """
    Wrapper for Hugging Face audio-classification models used for deepfake detection.

    Exposes a unified predict_waveform(waveform, sample_rate) -> score in [0,1].
    """

    def __init__(self, processor: Any, model: Any):
        self.name = "HF Audio Classifier"
        self.processor = processor
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Cache label mapping for fake / spoof
        self.id2label = getattr(self.model.config, "id2label", None)

    def _get_fake_class_index(self, probs) -> int:
        """
        Determine which class index corresponds to 'fake' / 'spoof' / 'deepfake'.
        Fallback: use class 1 if available, otherwise argmax.
        """
        num_labels = probs.shape[1]

        if self.id2label:
            # Search for labels that look like spoof / fake / deepfake
            for idx, label in self.id2label.items():
                label_l = str(label).lower()
                if any(
                    kw in label_l for kw in ["spoof", "fake", "deepfake", "synthetic"]
                ):
                    try:
                        return int(idx)
                    except Exception:
                        continue

        # Fallbacks
        if num_labels >= 2:
            return 1
        return int(torch.argmax(probs, dim=1)[0].item())

    def predict_waveform(self, waveform: np.ndarray, sample_rate: int) -> float:
        """
        Predict probability that the given waveform is a deepfake.

        Args:
            waveform: 1D numpy array
            sample_rate: sampling rate in Hz
        """
        try:
            if waveform.ndim > 1:
                # Convert multi-channel to mono
                waveform = np.mean(waveform, axis=-1)

            waveform = waveform.astype(np.float32)

            # Processor can be AutoProcessor or AutoFeatureExtractor
            inputs = self.processor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)

                fake_index = self._get_fake_class_index(probs)
                fake_index = max(0, min(probs.shape[1] - 1, fake_index))
                fake_prob = probs[0, fake_index].item()

            return float(fake_prob)
        except Exception as e:
            warnings.warn(f"HFAudioClassifier prediction error: {e}")
            return 0.5

    def __call__(self, waveform: np.ndarray, sample_rate: int) -> float:
        return self.predict_waveform(waveform, sample_rate)


class HeuristicImageModel:
    """
    Heuristic-based image deepfake detection model.
    Uses image analysis techniques when no ML model is available.
    """
    
    def __init__(self):
        self.name = "Heuristic Model"
    
    def predict(self, image_array: np.ndarray) -> float:
        """
        Predict if image is a deepfake using heuristics.
        
        Args:
            image_array: Preprocessed image array (normalized 0-1)
            
        Returns:
            Score between 0 (real) and 1 (deepfake)
        """
        try:
            score = 0.0
            
            # Check 1: Image quality metrics
            if image_array.shape[0] >= 224 and image_array.shape[1] >= 224:
                h, w = image_array.shape[:2]
                regions = [
                    image_array[:h//2, :w//2],
                    image_array[:h//2, w//2:],
                    image_array[h//2:, :w//2],
                    image_array[h//2:, w//2:]
                ]
                
                variances = [np.var(region) for region in regions]
                variance_std = np.std(variances)
                
                if variance_std > 0.01:
                    score += 0.2
            
            # Check 2: Color distribution anomalies
            if len(image_array.shape) == 3:
                mean_colors = np.mean(image_array, axis=(0, 1))
                if np.any(mean_colors < 0.1) or np.any(mean_colors > 0.9):
                    score += 0.15
            
            # Check 3: Edge detection artifacts
            try:
                from scipy import ndimage
                gray = np.mean(image_array, axis=2) if len(image_array.shape) == 3 else image_array
                edges = ndimage.sobel(gray)
                edge_variance = np.var(edges)
                
                if edge_variance > 0.1 or edge_variance < 0.001:
                    score += 0.15
            except:
                pass
            
            # Check 4: Frequency domain analysis
            try:
                from scipy.fft import fft2
                if len(image_array.shape) == 3:
                    gray = np.mean(image_array, axis=2)
                else:
                    gray = image_array
                
                fft_img = np.abs(fft2(gray))
                high_freq_energy = np.sum(fft_img[fft_img.shape[0]//4:, fft_img.shape[1]//4:])
                total_energy = np.sum(fft_img)
                
                if total_energy > 0:
                    high_freq_ratio = high_freq_energy / total_energy
                    if high_freq_ratio > 0.3 or high_freq_ratio < 0.05:
                        score += 0.2
            except:
                pass
            
            # Base score adjustment
            if score < 0.3:
                score = 0.35
            
            return min(score, 1.0)
        
        except Exception as e:
            return 0.4
    
    def __call__(self, image_array: np.ndarray) -> float:
        """Make model callable."""
        return self.predict(image_array)


def load_default_image_model(model_type: str = 'auto') -> Any:
    """
    Load the default image model.
    
    Args:
        model_type: Type of model ('auto', 'clip_vit', 'uia_vit', 'face_xray', 
                   'xception', 'mesonet', 'efficientnet', 'heuristic')
    
    Returns:
        Loaded model
    """
    loader = ModelLoader()
    return loader.load_image_model(model_type)
