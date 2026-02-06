"""
Robust Model Management System
"""
import torch
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized model loading and validation"""
    
    def __init__(self, models_dir: str = "/app/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Expected model checksums (update these when models change)
        self.expected_checksums = {
            "yolov8n-pose.pt": None,  # Will be calculated on first load
            "osnet_x0_25_msmt17.pt": None,
            "best_activity_model.pth": None,
            "best_accident_model.pth": None,
        }
    
    def load_model(self, model_name: str, model_class: Any = None, 
                   device: str = "cpu", **model_kwargs) -> Any:
        """
        Safely load and validate model
        """
        model_path = self.models_dir / model_name
        
        # Check if already loaded
        cache_key = f"{model_name}_{device}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        try:
            # Validate file exists
            if not model_path.exists():
                # Try alternative locations
                alt_paths = [
                    Path("/app") / model_name,
                    Path(".") / model_name,
                    Path("backend") / model_name
                ]
                
                for alt_path in alt_paths:
                    if alt_path.exists():
                        model_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"Model not found: {model_name}")
            
            # Validate file integrity
            if not self._validate_model_file(model_path, model_name):
                logger.warning(f"Model validation failed: {model_name}")
            
            # Load model based on type
            if model_name.endswith('.pt'):
                # YOLO or PyTorch model
                if 'yolo' in model_name.lower():
                    from ultralytics import YOLO
                    model = YOLO(str(model_path))
                else:
                    model = torch.load(model_path, map_location=device)
            
            elif model_name.endswith('.pth'):
                # Custom PyTorch state dict
                if model_class is None:
                    raise ValueError(f"model_class required for .pth files: {model_name}")
                
                model = model_class(**model_kwargs)
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
            
            else:
                raise ValueError(f"Unsupported model format: {model_name}")
            
            # Cache the loaded model
            self.loaded_models[cache_key] = model
            self.model_metadata[model_name] = {
                "loaded_at": datetime.now().isoformat(),
                "device": device,
                "file_size": model_path.stat().st_size,
                "checksum": self._calculate_checksum(model_path)
            }
            
            logger.info(f"Successfully loaded model: {model_name} on {device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            # Return mock model for graceful degradation
            return self._create_mock_model(model_name, model_class, **model_kwargs)
    
    def _validate_model_file(self, model_path: Path, model_name: str) -> bool:
        """Validate model file integrity"""
        try:
            # Check file size (models should be > 1MB)
            file_size = model_path.stat().st_size
            if file_size < 1024 * 1024:  # 1MB
                logger.warning(f"Model file suspiciously small: {file_size} bytes")
                return False
            
            # Calculate and verify checksum
            current_checksum = self._calculate_checksum(model_path)
            expected = self.expected_checksums.get(model_name)
            
            if expected is None:
                # First time loading, store checksum
                self.expected_checksums[model_name] = current_checksum
                logger.info(f"Stored checksum for {model_name}: {current_checksum[:8]}...")
                return True
            
            if current_checksum != expected:
                logger.error(f"Checksum mismatch for {model_name}")
                logger.error(f"Expected: {expected[:8]}..., Got: {current_checksum[:8]}...")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _create_mock_model(self, model_name: str, model_class: Any = None, **kwargs) -> Any:
        """Create mock model for graceful degradation"""
        logger.warning(f"Creating mock model for {model_name}")
        
        class MockModel:
            def __init__(self):
                self.is_mock = True
            
            def predict(self, *args, **kwargs):
                return None
            
            def __call__(self, *args, **kwargs):
                return None
            
            def eval(self):
                return self
            
            def to(self, device):
                return self
        
        return MockModel()
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get metadata about loaded model"""
        return self.model_metadata.get(model_name, {})
    
    def health_check(self) -> Dict:
        """Check health of all loaded models"""
        health_status = {
            "total_models": len(self.loaded_models),
            "models": {}
        }
        
        for model_name, metadata in self.model_metadata.items():
            health_status["models"][model_name] = {
                "status": "loaded",
                "device": metadata.get("device"),
                "loaded_at": metadata.get("loaded_at"),
                "file_size_mb": round(metadata.get("file_size", 0) / (1024*1024), 2)
            }
        
        return health_status

# Global model manager instance
model_manager = ModelManager()