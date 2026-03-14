"""
Image detector backed by umm-maybe/AI-image-detector.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from app.models.image.base import normalize_detector_result

logger = logging.getLogger(__name__)


class UMMMaybeDetector:
    """
    Uses: umm-maybe/AI-image-detector
    Task: image-classification
    """

    name = "umm_maybe"
    model_name = "umm-maybe/AI-image-detector"

    def __init__(self, device: Optional[str | int] = None):
        self.device = device
        self._pipe = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "transformers is required for UMMMaybeDetector. "
                "Install it in the same environment that runs uvicorn."
            ) from e

        logger.info(f"Loading UMMMaybeDetector ({self.model_name})...")
        start = time.time()
        kwargs: Dict[str, Any] = {"model": self.model_name}
        if self.device is not None:
            kwargs["device"] = self.device
        self._pipe = pipeline("image-classification", **kwargs)
        logger.info(f"✅ UMMMaybeDetector loaded in {(time.time() - start):.2f}s")

    def _parse(self, results: list[dict]) -> Tuple[float, float]:
        ai_keywords = ("ai", "generated", "synthetic", "fake", "artificial", "label_1")
        real_keywords = ("real", "human", "natural", "authentic", "original", "label_0")

        ai_score: Optional[float] = None
        real_score: Optional[float] = None

        for result in results:
            label = str(result.get("label", "")).lower()
            score = float(result.get("score", 0.0))
            if any(keyword in label for keyword in ai_keywords):
                ai_score = score
            elif any(keyword in label for keyword in real_keywords):
                real_score = score

        if ai_score is not None and real_score is None:
            real_score = 1.0 - ai_score
        if real_score is not None and ai_score is None:
            ai_score = 1.0 - real_score

        if ai_score is None and real_score is None:
            if results:
                ai_score = float(results[0].get("score", 0.0))
                real_score = 1.0 - ai_score
            else:
                ai_score = 0.0
                real_score = 1.0

        return float(ai_score), float(real_score)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if self._pipe is None:
            self._load_model()
        assert self._pipe is not None

        start = time.time()
        try:
            results = self._pipe(image)
            inference_ms = (time.time() - start) * 1000.0
            ai_score, real_score = self._parse(results)
            out = normalize_detector_result(
                detector_name=self.name,
                model_used=self.model_name,
                ai_probability=ai_score,
                real_probability=real_score,
                confidence=max(ai_score, real_score),
                inference_time_ms=inference_ms,
                raw_results=results,
            )
            out["is_ai_generated"] = ai_score > 0.5
            return out
        except Exception as e:
            logger.error(f"UMMMaybeDetector prediction failed: {e}")
            return {"success": False, "error": str(e), "detector": self.name, "model_used": self.model_name}
