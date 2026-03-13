from pydantic_settings import BaseSettings
from typing import Dict, List

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Content Detection ML Server"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]
    
    # Model paths (adjust as needed)
    MODEL_CACHE_DIR: str = "./models/cache"
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    
    # Device settings
    USE_GPU: bool = True
    GPU_DEVICE: int = 0

    # Image model selection / ensemble
    # You can override via env, e.g.
    # IMAGE_MODELS='["deepfake_v1","sdxl"]'
    # IMAGE_MODEL_WEIGHTS='{"deepfake_v1":0.7,"sdxl":0.3}'
    IMAGE_MODELS: List[str] = ["deepfake_v1", "sdxl"]
    IMAGE_MODEL_WEIGHTS: Dict[str, float] = {"deepfake_v1": 0.5, "sdxl": 0.5}
    RETURN_PER_MODEL: bool = True

    class Config:
        env_file = ".env"

# Create a global settings instance
settings = Settings()   # <-- This line is crucial!