"""
Image Detection API Routes
Endpoints for AI image detection
"""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.services.image_service import ImageDetectionService
from app.config import settings
from app.api.models.response_models import DetectionResponse, BatchDetectionResponse, HealthResponse
from functools import lru_cache

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
async def detect_image(
    file: UploadFile = File(..., description="Image file to analyze (jpg, png, webp)"),
    service: ImageDetectionService = Depends(get_image_service)
):
    """
    Detect if an uploaded image is AI-generated
    
    - Accepts: JPG, PNG, WEBP (up to 100MB)
    - Returns: Probability score and confidence level
    """
    try:
        # Run detection
        result = await service.detect_from_upload(file)
        
        # Return successful response
        return DetectionResponse(
            success=True,
            data=result
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions (they already have proper status codes)
        raise e
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.post("/detect-batch", response_model=BatchDetectionResponse)
async def detect_images_batch(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    service: ImageDetectionService = Depends(get_image_service)
):
    """
    Batch detection for multiple images
    
    - Accepts up to 10 images at once
    - Processes them sequentially (returns array of results)
    """
    try:
        # Limit batch size
        if len(files) > 10:
            raise HTTPException(
                status_code=400,
                detail="Too many files. Maximum 10 images per batch request."
            )
        
        # Run batch detection
        results = await service.detect_batch(files)
        
        return BatchDetectionResponse(
            success=True,
            data=results,
            total=len(results)
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch detection failed: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check(service: ImageDetectionService = Depends(get_image_service)):
    """
    Check if image detection service is operational
    """
    try:
        # Try to access detectors (will load if not loaded)
        _ = service.detectors
        return {
            "status": "healthy",
            "service": "image-detection",
            "models": settings.IMAGE_MODELS,
            "ensemble": {
                "combiner": "weighted_mean",
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