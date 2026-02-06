"""
Minimal configuration for RunPod ML Service.
This provides the same interface as backend/app/core/config.py
but with simpler environment-based configuration.
"""
import os


class Settings:
    """Minimal settings for RunPod environment."""
    
    # Face Recognition Settings
    FACE_DETECTOR_BACKEND: str = "retinaface"
    FACE_RECOGNITION_MODEL: str = "ArcFace"
    FACE_MATCH_THRESHOLD: float = 0.45
    EMBEDDINGS_DB_PATH: str = "/app/employee_embeddings.pkl"
    MIN_FACE_SIZE: int = 40
    
    # ML Model Paths
    PERSON_ID_MODEL_PATH: str = "/app/models/person_identification.pth"
    ACTIVITY_MODEL_PATH: str = "/app/models/best_activity_model.pth"
    ACCIDENT_MODEL_PATH: str = "/app/models/best_accident_model.pth"
    THEFT_MODEL_PATH: str = "/app/models/theft_detector.pth"
    
    def __init__(self):
        # Override from environment if available
        self.FACE_MATCH_THRESHOLD = float(os.environ.get(
            "FACE_MATCH_THRESHOLD", self.FACE_MATCH_THRESHOLD
        ))
        self.MIN_FACE_SIZE = int(os.environ.get(
            "MIN_FACE_SIZE", self.MIN_FACE_SIZE
        ))


settings = Settings()
