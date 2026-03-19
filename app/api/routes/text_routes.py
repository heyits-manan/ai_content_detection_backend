from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.models.request_models import TextDetectionRequest
from app.api.models.response_models import HealthResponse, TextDetectionResponse
from app.config import settings
from app.core.rate_limit import limiter
from app.services.text_service import TextDetectionService

router = APIRouter()


@lru_cache(maxsize=1)
def _get_text_service_singleton() -> TextDetectionService:
    return TextDetectionService()


async def get_text_service():
    return _get_text_service_singleton()


@router.post("/detect", response_model=TextDetectionResponse)
@limiter.limit(settings.TEXT_DETECT_RATE_LIMIT)
async def detect_text(
    request: Request,
    payload: TextDetectionRequest,
    service: TextDetectionService = Depends(get_text_service),
):
    result = await service.detect_text(payload.text)
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Text detection failed"))
    return TextDetectionResponse(success=True, data=result)


@router.get("/health", response_model=HealthResponse)
@limiter.limit(settings.TEXT_HEALTH_RATE_LIMIT)
async def health_check(request: Request, service: TextDetectionService = Depends(get_text_service)):
    try:
        return {
            "status": "healthy",
            "service": "text-detection",
            "models": settings.TEXT_MODELS,
            "ensemble": {
                "combiner": "mean_of_successful_models",
                "weights": settings.TEXT_MODEL_WEIGHTS,
                "return_per_model": True,
            },
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "text-detection",
            "error": str(e),
        }
