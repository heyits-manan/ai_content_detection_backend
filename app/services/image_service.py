"""
Image Detection Service
Handles image validation, preprocessing, and detection orchestration
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from fastapi import UploadFile, HTTPException

from app.config import settings
from app.models.image.base import clamp01
from app.models.image.registry import get_detectors

logger = logging.getLogger(__name__)


def _normalize_weights(
    keys: List[str],
    weights: Dict[str, float],
) -> Dict[str, float]:
    w: Dict[str, float] = {}
    for k in keys:
        w[k] = float(weights.get(k, 1.0))
    total = sum(max(0.0, v) for v in w.values())
    if total <= 0:
        # fallback to equal weights
        eq = 1.0 / max(1, len(keys))
        return {k: eq for k in keys}
    return {k: max(0.0, v) / total for k, v in w.items()}


def combine_ai_probabilities(
    per_model: List[Dict[str, Any]],
    weights: Dict[str, float],
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Combine per-model outputs into a single ai_probability.

    - Excludes failed models (success != True)
    - Renormalizes weights over remaining models
    """
    successes = [r for r in per_model if r.get("success") is True]
    if not successes:
        return 0.0, per_model

    keys = [str(r.get("detector") or r.get("model_used") or "unknown") for r in successes]
    w_norm = _normalize_weights(keys, weights)

    ai = 0.0
    for r in successes:
        k = str(r.get("detector") or r.get("model_used") or "unknown")
        ai += float(w_norm.get(k, 0.0)) * float(r.get("ai_probability", 0.0))
    return clamp01(ai), per_model


class ImageDetectionService:
    """
    Service for detecting AI-generated images
    """
    
    def __init__(self):
        """Initialize the service and load the detector"""
        self.detectors = None
        self._load_detectors()
    
    def _load_detectors(self):
        """Lazy-load detectors. Uses a process-level cache under the hood."""
        if self.detectors is None:
            logger.info("Initializing image detector service...")
            keys = tuple(settings.IMAGE_MODELS)
            self.detectors = get_detectors(keys)
    
    async def validate_image(self, file: UploadFile) -> tuple:
        """
        Validate uploaded image file
        
        Args:
            file: Uploaded file from FastAPI
            
        Returns:
            Tuple of (is_valid, error_message, file_path)
        """
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            return False, f"File type {file_ext} not allowed. Allowed: {settings.ALLOWED_IMAGE_EXTENSIONS}", None
        
        # Check file size (read first chunk to check)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > settings.MAX_UPLOAD_SIZE:
            return False, f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / (1024*1024)}MB", None
        
        # Save temporarily for processing
        temp_path = os.path.join(settings.UPLOAD_DIR, f"temp_{file.filename}")
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        try:
            # Save file
            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            
            # Verify it's a valid image
            try:
                with Image.open(temp_path) as img:
                    # Just verify it opens, don't need to do anything else
                    logger.info(f"Image validated: {file.filename}, size: {img.size}, format: {img.format}")
            except Exception as e:
                os.remove(temp_path)
                return False, f"Invalid image file: {str(e)}", None
            
            return True, None, temp_path
            
        except Exception as e:
            logger.error(f"Error saving/validating image: {e}")
            return False, f"Error processing upload: {str(e)}", None
    
    async def detect_from_file(self, file_path: str) -> dict:
        """
        Run detection on an image file
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Detection results dictionary
        """
        try:
            # Load image
            image = Image.open(file_path)
            
            # Ensure detectors are loaded
            self._load_detectors()
            assert self.detectors is not None

            per_model: List[Dict[str, Any]] = []
            for detector in self.detectors:
                try:
                    per_model.append(detector.predict(image))
                except Exception as e:
                    per_model.append(
                        {
                            "success": False,
                            "detector": getattr(detector, "name", "unknown"),
                            "model_used": getattr(detector, "model_name", "unknown"),
                            "error": str(e),
                        }
                    )

            # Combine (weighted mean; defaults to equal weights)
            ai_p, per_model = combine_ai_probabilities(per_model, settings.IMAGE_MODEL_WEIGHTS)
            
            # Calculate flat average just in case weights are unbalanced
            successful_models = [r for r in per_model if r.get("success") is True]
            if successful_models:
                avg_ai = sum(float(r.get("ai_probability", 0.0)) for r in successful_models) / len(successful_models)
                avg_real = sum(float(r.get("real_probability", 0.0)) for r in successful_models) / len(successful_models)
            else:
                avg_ai = ai_p
                avg_real = 1.0 - ai_p
                
            out: Dict[str, Any] = {
                "success": True,
                "ai_probability": ai_p,
                "real_probability": float(1.0 - ai_p),
                "average_ai_probability": avg_ai,
                "average_real_probability": avg_real,
                "is_ai_generated": avg_ai > 0.5,
                "confidence": float(max(ai_p, 1.0 - ai_p)),
                "models_used": [r.get("detector") for r in successful_models],
                "filename": os.path.basename(file_path),
            }
            if settings.RETURN_PER_MODEL:
                out["per_model"] = per_model

            # If none succeeded, surface as failure
            if not out["models_used"]:
                return {
                    "success": False,
                    "error": "All detectors failed",
                    "filename": os.path.basename(file_path),
                    "per_model": per_model,
                }

            return out
            
        except Exception as e:
            logger.error(f"Detection failed for {file_path}: {e}")
            return {
                "success": False,
                "error": f"Detection failed: {str(e)}",
                "filename": os.path.basename(file_path)
            }
    
    async def detect_from_upload(self, file: UploadFile) -> dict:
        """
        Complete pipeline: validate, save, detect, cleanup
        
        Args:
            file: Uploaded file from FastAPI
            
        Returns:
            Detection results dictionary
        """
        temp_path = None
        
        try:
            # Validate
            is_valid, error_msg, temp_path = await self.validate_image(file)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Detect
            result = await self.detect_from_file(temp_path)
            
            if not result.get("success", False):
                raise HTTPException(status_code=500, detail=result.get("error", "Detection failed"))
            
            return result
            
        finally:
            # Cleanup temp file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"Cleaned up temp file: {temp_path}")
    
    async def detect_batch(self, files: list[UploadFile]) -> list[dict]:
        """
        Batch detection for multiple images
        
        Args:
            files: List of uploaded files
            
        Returns:
            List of detection results
        """
        results = []
        for file in files:
            try:
                result = await self.detect_from_upload(file)
                results.append(result)
            except Exception as e:
                results.append({
                    "success": False,
                    "filename": file.filename,
                    "error": str(e)
                })
        return results