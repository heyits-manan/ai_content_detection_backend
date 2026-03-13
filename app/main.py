from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes import image_routes  # Add this import
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

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(image_routes.router, prefix="/api/v1/image", tags=["image"])
# app.include_router(video_routes.router, prefix="/api/v1/video", tags=["video"])
# app.include_router(audio_routes.router, prefix="/api/v1/audio", tags=["audio"])
# app.include_router(text_routes.router, prefix="/api/v1/text", tags=["text"])

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
        "services": ["image"]  # Add others as you build them
    }