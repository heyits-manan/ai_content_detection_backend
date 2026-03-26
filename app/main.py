from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.middleware import SlowAPIMiddleware

from app.config import settings
from app.api.routes import audio_routes, image_routes, job_routes, text_routes, video_routes
from app.core.rate_limit import limiter
from app.core.exceptions import (
    add_request_id_middleware,
    app_error_handler,
    http_exception_handler,
    rate_limit_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.core.exceptions import AppError
import logging
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from slowapi.errors import RateLimitExceeded

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
app.middleware("http")(add_request_id_middleware)
app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

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
app.include_router(job_routes.router, prefix=settings.API_V1_STR, tags=["jobs"])
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
