"""
Video Detection API Routes
"""

from functools import lru_cache
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.models.response_models import HealthResponse, VideoDetectionResponse
from app.config import settings
from app.core.rate_limit import limiter
from app.services.video_service import VideoDetectionService

router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_video_service_singleton() -> VideoDetectionService:
    return VideoDetectionService()


async def get_video_service() -> VideoDetectionService:
    return _get_video_service_singleton()


@router.post("/detect", response_model=VideoDetectionResponse)
@limiter.limit(settings.VIDEO_DETECT_RATE_LIMIT)
async def detect_video(
    request: Request,
    file: UploadFile = File(..., description="Video file to analyze"),
    num_frames: int = Form(settings.VIDEO_DEFAULT_NUM_FRAMES, ge=1, le=256),
    service: VideoDetectionService = Depends(get_video_service),
):
    logger.info(
        "Video detect request received: method=%s path=%s client=%s filename=%s content_type=%s num_frames=%s content_length=%s",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
        file.filename,
        file.content_type,
        num_frames,
        request.headers.get("content-length"),
    )
    try:
        result = await service.detect_from_upload(file=file, num_frames=num_frames)
        logger.info(
            "Video detect request completed: filename=%s ai_probability=%.4f num_frames_used=%s",
            file.filename,
            result["ai_probability"],
            result["num_frames_used"],
        )
        return result
    except HTTPException as exc:
        logger.warning(
            "Video detect request failed: filename=%s status=%s detail=%s",
            file.filename,
            exc.status_code,
            exc.detail,
        )
        raise
    except Exception as exc:
        logger.exception("Unexpected video detect request error for filename=%s", file.filename)
        raise HTTPException(status_code=500, detail=f"Unexpected video detection error: {exc}") from exc


@router.get("/health", response_model=HealthResponse)
@limiter.limit(settings.VIDEO_HEALTH_RATE_LIMIT)
async def health_check(request: Request) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="video-detection",
        models=settings.IMAGE_MODELS,
    )
