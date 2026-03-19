from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from app.config import settings
from app.api.routes import audio_routes, image_routes, text_routes, video_routes
from app.core.rate_limit import limiter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Content Detection API",
    description="ML inference server for detecting AI-generated content",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)

# Include routers
app.include_router(image_routes.router, prefix="/api/v1/image", tags=["image"])
app.include_router(text_routes.router, prefix="/api/v1/text", tags=["text"])
app.include_router(video_routes.router, prefix="/video", tags=["video"])
app.include_router(audio_routes.router, prefix="/api/v1/audio", tags=["audio"])

@app.get("/")
async def root():
    return {
        "message": "AI Content Detection API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": ["image", "text", "video", "audio"]
    }
