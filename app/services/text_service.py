from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from app.config import settings
from app.models.text.roberta_detector import (
    HelloSimpleAIRobertaDetector,
    OpenAIRobertaDetector,
)

logger = logging.getLogger(__name__)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _normalize_weights(keys: List[str], weights: Dict[str, float]) -> Dict[str, float]:
    current = {key: float(weights.get(key, 1.0)) for key in keys}
    total = sum(max(0.0, value) for value in current.values())
    if total <= 0:
        equal = 1.0 / max(1, len(keys))
        return {key: equal for key in keys}
    return {key: max(0.0, value) / total for key, value in current.items()}


def combine_text_probabilities(per_model: List[Dict[str, Any]], weights: Dict[str, float]) -> float:
    successes = [result for result in per_model if result.get("success") is True]
    if not successes:
        return 0.0

    keys = [str(result.get("detector") or result.get("model_used") or "unknown") for result in successes]
    normalized = _normalize_weights(keys, weights)
    weighted = 0.0
    for result in successes:
        key = str(result.get("detector") or result.get("model_used") or "unknown")
        weighted += float(normalized.get(key, 0.0)) * float(result.get("ai_probability", 0.0))
    return _clamp01(weighted)


def average_text_probabilities(per_model: List[Dict[str, Any]]) -> Tuple[float, float]:
    successes = [result for result in per_model if result.get("success") is True]
    if not successes:
        return 0.0, 1.0

    avg_ai = sum(float(result.get("ai_probability", 0.0)) for result in successes) / len(successes)
    avg_human = sum(float(result.get("human_probability", 0.0)) for result in successes) / len(successes)
    return _clamp01(avg_ai), _clamp01(avg_human)


@lru_cache(maxsize=1)
def get_text_detectors(keys: tuple[str, ...]):
    factories = {
        "openai_roberta": OpenAIRobertaDetector,
        "hello_simpleai_roberta": HelloSimpleAIRobertaDetector,
    }
    detectors = []
    for key in keys:
        if key not in factories:
            raise ValueError(f"Unknown text detector '{key}'. Available: {sorted(factories.keys())}")
        logger.info(f"Loading text detector '{key}'...")
        detectors.append(factories[key]())
    return detectors


class TextDetectionService:
    def __init__(self):
        self.detectors = None

    def _load_detectors(self) -> None:
        if self.detectors is None:
            self.detectors = get_text_detectors(tuple(settings.TEXT_MODELS))

    async def detect_text(self, text: str) -> Dict[str, Any]:
        self._load_detectors()
        assert self.detectors is not None

        per_model: List[Dict[str, Any]] = []
        for detector in self.detectors:
            try:
                per_model.append(detector.predict(text))
            except Exception as e:
                per_model.append(
                    {
                        "success": False,
                        "detector": getattr(detector, "name", "unknown"),
                        "model_used": getattr(detector, "model_name", "unknown"),
                        "error": str(e),
                    }
                )

        successful_models = [result for result in per_model if result.get("success") is True]
        if not successful_models:
            return {
                "success": False,
                "error": "All text detectors failed",
                "text_length": len(text),
                "per_model": per_model,
            }

        weighted_ai = combine_text_probabilities(per_model, settings.TEXT_MODEL_WEIGHTS)
        avg_ai, avg_human = average_text_probabilities(per_model)
        return {
            "success": True,
            "ai_probability": avg_ai,
            "human_probability": avg_human,
            "weighted_ai_probability": weighted_ai,
            "average_ai_probability": avg_ai,
            "average_human_probability": avg_human,
            "is_ai_generated": avg_ai > 0.5,
            "confidence": max(avg_ai, avg_human),
            "models_used": [result.get("detector") for result in successful_models],
            "text_length": len(text),
            "per_model": per_model,
        }
