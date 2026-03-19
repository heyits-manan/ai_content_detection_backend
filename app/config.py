import os

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
    HF_HOME: str = "/opt/huggingface/hub" #change this to .hf-cache/ when running locally
    
    HF_LOCAL_FILES_ONLY: bool = True
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_VIDEO_UPLOAD_SIZE: int = 250 * 1024 * 1024  # 250MB
    MAX_AUDIO_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    ALLOWED_AUDIO_EXTENSIONS: List[str] = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
    ALLOWED_VIDEO_CONTENT_TYPES: List[str] = [
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/x-matroska",
        "video/webm",
        "application/mp4",
        "application/octet-stream",
    ]
    ALLOWED_AUDIO_CONTENT_TYPES: List[str] = [
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/x-flac",
        "audio/mp4",
        "audio/x-m4a",
        "audio/aac",
        "audio/ogg",
        "application/octet-stream",
    ]
    VIDEO_UPLOAD_CHUNK_SIZE: int = 1024 * 1024
    AUDIO_UPLOAD_CHUNK_SIZE: int = 1024 * 1024
    
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
    VIDEO_DEFAULT_NUM_FRAMES: int = 16
    VIDEO_MAX_WORKERS: int = max(1, min(4, os.cpu_count() or 1))
    VIDEO_INFERENCE_TIMEOUT_SECONDS: float = 180.0
    VIDEO_DETECT_RATE_LIMIT: str = "5/minute"
    VIDEO_HEALTH_RATE_LIMIT: str = "60/minute"
    AUDIO_MODEL: str = "garystafford/wav2vec2-deepfake-voice-detector"
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHUNK_SECONDS: float = 5.0
    AUDIO_CHUNK_OVERLAP_SECONDS: float = 1.0
    AUDIO_MIN_CHUNK_SECONDS: float = 2.5
    AUDIO_DETECT_RATE_LIMIT: str = "10/minute"
    AUDIO_HEALTH_RATE_LIMIT: str = "60/minute"
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
