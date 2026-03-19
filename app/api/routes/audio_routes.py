from functools import lru_cache

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.api.models.response_models import AudioDetectionResponse, HealthResponse
from app.config import settings
from app.core.rate_limit import limiter
from app.services.audio_service import AudioDetectionService

router = APIRouter()


@lru_cache(maxsize=1)
def _get_audio_service_singleton() -> AudioDetectionService:
    return AudioDetectionService()


async def get_audio_service() -> AudioDetectionService:
    return _get_audio_service_singleton()


@router.post("/detect", response_model=AudioDetectionResponse)
@limiter.limit(settings.AUDIO_DETECT_RATE_LIMIT)
async def detect_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to analyze"),
    service: AudioDetectionService = Depends(get_audio_service),
):
    result = await service.detect_from_upload(file)
    return AudioDetectionResponse(success=True, data=result)


@router.get("/health", response_model=HealthResponse)
@limiter.limit(settings.AUDIO_HEALTH_RATE_LIMIT)
async def health_check(request: Request) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="audio-detection",
        models=[settings.AUDIO_MODEL],
    )
