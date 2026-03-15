"""
Text detectors backed by Hugging Face Transformers pipelines.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from app.models.hf_loader import load_text_pipeline

logger = logging.getLogger(__name__)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class BaseRobertaTextDetector:
    name: str
    model_name: str

    def __init__(self, device: Optional[str | int] = None):
        self.device = device
        self._pipe = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info(f"Loading {self.name} ({self.model_name})...")
        start = time.time()
        self._pipe = load_text_pipeline(self.model_name, device=self.device)
        logger.info(f"Loaded {self.name} in {(time.time() - start):.2f}s")

    def _parse(self, results: list[dict]) -> Tuple[float, float]:
        ai_keywords = ("fake", "ai", "generated", "machine", "label_1")
        human_keywords = ("real", "human", "original", "human-written", "label_0")

        ai_score: Optional[float] = None
        human_score: Optional[float] = None

        for result in results:
            label = str(result.get("label", "")).lower()
            score = float(result.get("score", 0.0))
            if any(keyword in label for keyword in ai_keywords):
                ai_score = score
            elif any(keyword in label for keyword in human_keywords):
                human_score = score

        if ai_score is not None and human_score is None:
            human_score = 1.0 - ai_score
        if human_score is not None and ai_score is None:
            ai_score = 1.0 - human_score

        if ai_score is None and human_score is None:
            if results:
                ai_score = float(results[0].get("score", 0.0))
                human_score = 1.0 - ai_score
            else:
                ai_score = 0.0
                human_score = 1.0

        return _clamp01(float(ai_score)), _clamp01(float(human_score))

    def predict(self, text: str) -> Dict[str, Any]:
        if self._pipe is None:
            self._load_model()
        assert self._pipe is not None

        start = time.time()
        try:
            results = self._pipe(text)
            inference_ms = (time.time() - start) * 1000.0
            ai_score, human_score = self._parse(results)
            return {
                "success": True,
                "detector": self.name,
                "model_used": self.model_name,
                "ai_probability": ai_score,
                "human_probability": human_score,
                "confidence": max(ai_score, human_score),
                "inference_time_ms": inference_ms,
                "raw_results": results,
                "is_ai_generated": ai_score > 0.5,
            }
        except Exception as e:
            logger.error(f"{self.name} prediction failed: {e}")
            return {
                "success": False,
                "detector": self.name,
                "model_used": self.model_name,
                "error": str(e),
            }


class OpenAIRobertaDetector(BaseRobertaTextDetector):
    name = "openai_roberta"
    model_name = "openai-community/roberta-base-openai-detector"


class HelloSimpleAIRobertaDetector(BaseRobertaTextDetector):
    name = "hello_simpleai_roberta"
    model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
