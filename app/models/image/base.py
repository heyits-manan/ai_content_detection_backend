"""
Shared interfaces/types for image detectors.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from PIL import Image


class BaseImageDetector(Protocol):
    """
    Minimal contract the service relies on.
    """

    name: str
    model_name: str

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Returns a dict with at least:
        - success: bool
        - ai_probability: float (0..1)
        - model_used: str
        """


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def normalize_detector_result(
    *,
    detector_name: str,
    model_used: str,
    ai_probability: float,
    real_probability: Optional[float] = None,
    confidence: Optional[float] = None,
    inference_time_ms: Optional[float] = None,
    raw_results: Any = None,
) -> Dict[str, Any]:
    ai_p = clamp01(float(ai_probability))
    real_p = clamp01(float(1.0 - ai_p) if real_probability is None else float(real_probability))
    conf = float(max(ai_p, real_p)) if confidence is None else float(confidence)

    out: Dict[str, Any] = {
        "success": True,
        "detector": detector_name,
        "model_used": model_used,
        "ai_probability": ai_p,
        "real_probability": real_p,
        "confidence": conf,
    }
    if inference_time_ms is not None:
        out["inference_time_ms"] = float(inference_time_ms)
    if raw_results is not None:
        out["raw_results"] = raw_results
    return out
