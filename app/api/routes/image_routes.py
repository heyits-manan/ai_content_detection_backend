"""
Image Detection API Routes
Endpoints for AI image detection
"""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request

from app.services.image_service import ImageDetectionService
from app.config import settings
from app.api.models.response_models import DetectionResponse, BatchDetectionResponse, HealthResponse
from functools import lru_cache
from app.core.rate_limit import limiter

# Create router
router = APIRouter()

# Dependency to get image service
@lru_cache(maxsize=1)
def _get_image_service_singleton() -> ImageDetectionService:
    return ImageDetectionService()

async def get_image_service():
    """Dependency injection for ImageDetectionService"""
    return _get_image_service_singleton()

@router.post("/detect", response_model=DetectionResponse)
@limiter.limit(settings.IMAGE_DETECT_RATE_LIMIT)
async def detect_image(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze (jpg, png, webp)"),
    service: ImageDetectionService = Depends(get_image_service)
):
    """
    Detect if an uploaded image is AI-generated
    
    - Accepts: JPG, PNG, WEBP (up to 100MB)
    - Returns: Probability score and confidence level
    """
    result = await service.detect_from_upload(file)
    return DetectionResponse(
        success=True,
        data=result
    )

@router.post("/detect-batch", response_model=BatchDetectionResponse)
@limiter.limit(settings.IMAGE_BATCH_RATE_LIMIT)
async def detect_images_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="Multiple image files"),
    service: ImageDetectionService = Depends(get_image_service)
):
    """
    Batch detection for multiple images
    
    - Accepts up to 10 images at once
    - Processes them sequentially (returns array of results)
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Too many files. Maximum 10 images per batch request."
        )

    results = await service.detect_batch(files)

    return BatchDetectionResponse(
        success=True,
        data=results,
        total=len(results)
    )

@router.get("/health", response_model=HealthResponse)
@limiter.limit(settings.IMAGE_HEALTH_RATE_LIMIT)
async def health_check(request: Request, service: ImageDetectionService = Depends(get_image_service)):
    """
    Check if image detection service is operational
    """
    try:
        return {
            "status": "healthy",
            "service": "image-detection",
            "models": settings.IMAGE_MODELS,
            "ensemble": {
                "combiner": "mean_of_successful_models",
                "weights": settings.IMAGE_MODEL_WEIGHTS,
                "return_per_model": settings.RETURN_PER_MODEL,
            },
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "image-detection",
            "error": str(e)
        }
