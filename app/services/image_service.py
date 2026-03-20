"""
Image Detection Service
Handles image validation, preprocessing, and detection orchestration
"""

import os
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from fastapi import UploadFile

from app.config import settings
from app.core.exceptions import BadRequestError, InferenceFailedError
from app.core.file_handler import remove_file_if_exists, resolve_upload_suffix, save_upload_to_temp
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


def average_ai_probabilities(per_model: List[Dict[str, Any]]) -> Tuple[float, float]:
    successes = [r for r in per_model if r.get("success") is True]
    if not successes:
        return 0.0, 1.0

    avg_ai = sum(float(r.get("ai_probability", 0.0)) for r in successes) / len(successes)
    avg_real = sum(float(r.get("real_probability", 0.0)) for r in successes) / len(successes)
    return clamp01(avg_ai), clamp01(avg_real)


def _frame_to_pil_image(frame: np.ndarray | Image.Image) -> Image.Image:
    """
    Convert either a PIL image or an OpenCV-style ndarray into RGB PIL format.
    """
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy array or PIL image")

    if frame.ndim == 2:
        return Image.fromarray(frame).convert("RGB")

    if frame.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    channels = frame.shape[2]
    if channels == 3:
        return Image.fromarray(frame[:, :, ::-1]).convert("RGB")
    if channels == 4:
        return Image.fromarray(frame[:, :, [2, 1, 0, 3]], mode="RGBA").convert("RGB")
    raise ValueError(f"Unsupported frame channel count: {channels}")


def run_image_ensemble(frame: np.ndarray | Image.Image) -> float:
    """
    Run the existing image ensemble on a single frame and return the AI score.

    `np.ndarray` inputs are treated as OpenCV BGR/BGRA frames.
    """
    image = _frame_to_pil_image(frame)
    detectors = get_detectors(tuple(settings.IMAGE_MODELS))

    per_model: List[Dict[str, Any]] = []
    for detector in detectors:
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

    if not any(result.get("success") is True for result in per_model):
        raise RuntimeError("All image detectors failed")

    avg_ai, _ = average_ai_probabilities(per_model)
    return avg_ai


def ensure_image_ensemble_loaded() -> None:
    """
    Warm the image ensemble into process memory before parallel inference starts.
    """
    get_detectors(tuple(settings.IMAGE_MODELS))


class ImageDetectionService:
    """
    Service for detecting AI-generated images
    """
    
    def __init__(self):
        """Initialize the service and load the detector"""
        self.detectors = None
    
    def _load_detectors(self):
        """Lazy-load detectors. Uses a process-level cache under the hood."""
        if self.detectors is None:
            logger.info("Initializing image detector service...")
            keys = tuple(settings.IMAGE_MODELS)
            self.detectors = get_detectors(keys)

    async def save_validated_upload(self, file: UploadFile) -> str:
        suffix = resolve_upload_suffix(
            filename=file.filename,
            content_type=file.content_type,
            allowed_extensions=settings.ALLOWED_IMAGE_EXTENSIONS,
            allowed_content_types=settings.ALLOWED_IMAGE_CONTENT_TYPES,
            generic_prefix="image/",
        )
        if not suffix:
            raise BadRequestError(
                "Invalid image upload.",
                details={
                    "filename": file.filename or "<missing>",
                    "content_type": (file.content_type or "<missing>").lower(),
                    "allowed_extensions": settings.ALLOWED_IMAGE_EXTENSIONS,
                },
            )

        try:
            saved_upload = await save_upload_to_temp(
                file=file,
                upload_dir=settings.UPLOAD_DIR,
                prefix="image_",
                suffix=suffix,
                chunk_size=settings.IMAGE_UPLOAD_CHUNK_SIZE,
                max_size_bytes=settings.MAX_UPLOAD_SIZE,
            )
        except ValueError as exc:
            raise BadRequestError(str(exc)) from exc

        try:
            with Image.open(saved_upload.temp_path) as img:
                img.verify()
            with Image.open(saved_upload.temp_path) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    raise BadRequestError("Invalid image dimensions")
                if width * height > settings.MAX_IMAGE_PIXELS:
                    raise BadRequestError(
                        f"Image is too large to process. Max pixels: {settings.MAX_IMAGE_PIXELS}",
                        details={"width": width, "height": height},
                    )
                logger.info(
                    "Image validated: filename=%s size=%sx%s format=%s bytes=%s",
                    file.filename,
                    width,
                    height,
                    img.format,
                    saved_upload.size_bytes,
                )
        except BadRequestError:
            remove_file_if_exists(saved_upload.temp_path)
            raise
        except Exception as exc:
            remove_file_if_exists(saved_upload.temp_path)
            raise BadRequestError(f"Invalid image file: {exc}") from exc

        return saved_upload.temp_path
    
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
            image = Image.open(file_path).convert("RGB")
            
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

            successful_models = [r for r in per_model if r.get("success") is True]
            weighted_ai, per_model = combine_ai_probabilities(per_model, settings.IMAGE_MODEL_WEIGHTS)
            avg_ai, avg_real = average_ai_probabilities(per_model)
                
            out: Dict[str, Any] = {
                "success": True,
                "ai_probability": avg_ai,
                "real_probability": avg_real,
                "average_ai_probability": avg_ai,
                "average_real_probability": avg_real,
                "is_ai_generated": avg_ai > 0.5,
                "confidence": float(max(avg_ai, avg_real)),
                "weighted_ai_probability": weighted_ai,
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
            temp_path = await self.save_validated_upload(file)
            result = await self.detect_from_file(temp_path)
            
            if not result.get("success", False):
                raise InferenceFailedError(result.get("error", "Detection failed"))
            
            return result
            
        finally:
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
