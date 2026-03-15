from pydantic_settings import BaseSettings
from typing import Dict, List

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Content Detection ML Server"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        # "http://localhost:3000",
        # "http://127.0.0.1:3000",
        # "http://localhost:3001",
        # "http://127.0.0.1:3001",
        "*"
    ]
    
    # Model paths (adjust as needed)
    MODEL_CACHE_DIR: str = "./models/cache"
    HF_HOME: str = "./.hf-cache"
    HF_LOCAL_FILES_ONLY: bool = True
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    
    # Device settings
    USE_GPU: bool = True
    GPU_DEVICE: int = 0

    # Image model selection / ensemble
    # You can override via env, e.g.
    # IMAGE_MODELS='["deepfake_v1","sdxl","umm_maybe"]'
    # IMAGE_MODEL_WEIGHTS='{"deepfake_v1":1.0,"sdxl":1.0,"umm_maybe":1.0}'
    IMAGE_MODELS: List[str] = ["deepfake_v1", "sdxl", "umm_maybe"]
    IMAGE_MODEL_WEIGHTS: Dict[str, float] = {
        "deepfake_v1": 1.0,
        "sdxl": 1.0,
        "umm_maybe": 1.0,
    }
    RETURN_PER_MODEL: bool = True
    DEFAULT_RATE_LIMIT: str = "30/minute"
    IMAGE_DETECT_RATE_LIMIT: str = "10/minute"
    IMAGE_BATCH_RATE_LIMIT: str = "3/minute"
    IMAGE_HEALTH_RATE_LIMIT: str = "60/minute"
    TEXT_MODELS: List[str] = ["openai_roberta", "hello_simpleai_roberta"]
    TEXT_MODEL_WEIGHTS: Dict[str, float] = {
        "openai_roberta": 1.0,
        "hello_simpleai_roberta": 1.0,
    }
    TEXT_DETECT_RATE_LIMIT: str = "20/minute"
    TEXT_HEALTH_RATE_LIMIT: str = "60/minute"

    class Config:
        env_file = ".env"

# Create a global settings instance
settings = Settings()   # <-- This line is crucial!
