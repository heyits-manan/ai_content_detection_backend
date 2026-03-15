"""
Deepfake detector backed by a Hugging Face Transformers pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from app.models.image.base import normalize_detector_result
from app.models.hf_loader import load_image_pipeline

logger = logging.getLogger(__name__)


class DeepfakeDetectorV1:
    """
    Uses: prithivMLmods/deepfake-detector-model-v1
    Task: image-classification
    """

    name = "deepfake_v1"
    model_name = "prithivMLmods/deepfake-detector-model-v1"

    def __init__(self, device: Optional[str | int] = None):
        """
        device:
        - None: let transformers decide (usually CPU)
        - int: GPU index (cuda) for pipeline
        - str: accepted by pipeline in some versions; keep optional
        """
        self.device = device
        self._pipe = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info(f"Loading DeepfakeDetectorV1 ({self.model_name})...")
        start = time.time()
        self._pipe = load_image_pipeline(self.model_name, device=self.device)
        logger.info(f"✅ DeepfakeDetectorV1 loaded in {(time.time() - start):.2f}s")

    def _parse(self, results: list[dict]) -> Tuple[float, float]:
        """
        Map label/score pairs into (ai_probability, real_probability).
        Treat "fake/deepfake/ai/generated" as AI.
        """
        ai_keywords = ("fake", "deepfake", "ai", "generated", "synthetic")
        real_keywords = ("real", "authentic", "human", "natural", "genuine")

        ai_score: Optional[float] = None
        real_score: Optional[float] = None

        for r in results:
            label = str(r.get("label", "")).lower()
            score = float(r.get("score", 0.0))
            if any(k in label for k in ai_keywords):
                ai_score = score
            elif any(k in label for k in real_keywords):
                real_score = score

        # Fall back: if there are 2 classes and only one identified, infer the other.
        if ai_score is not None and real_score is None:
            real_score = 1.0 - ai_score
        if real_score is not None and ai_score is None:
            ai_score = 1.0 - real_score

        # Final fallback: if unknown labels, use max score as "AI probability" only if label looks AI;
        # otherwise return score for first entry and infer complement.
        if ai_score is None and real_score is None:
            if results:
                top = max(results, key=lambda x: float(x.get("score", 0.0)))
                top_label = str(top.get("label", "")).lower()
                top_score = float(top.get("score", 0.0))
                if any(k in top_label for k in ai_keywords):
                    ai_score = top_score
                    real_score = 1.0 - ai_score
                elif any(k in top_label for k in real_keywords):
                    real_score = top_score
                    ai_score = 1.0 - real_score
                else:
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
            logger.error(f"DeepfakeDetectorV1 prediction failed: {e}")
            return {"success": False, "error": str(e), "detector": self.name, "model_used": self.model_name}
