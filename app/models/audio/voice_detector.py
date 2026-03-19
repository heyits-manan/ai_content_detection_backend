from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from app.config import settings
from app.models.audio.base import BaseAudioDetector
from app.models.hf_loader import load_audio_classification_components

logger = logging.getLogger(__name__)


class VoiceDeepfakeDetector(BaseAudioDetector):
    """
    Audio deepfake detector backed by garystafford/wav2vec2-deepfake-voice-detector.
    """

    name = "wav2vec2_deepfake_voice"

    def __init__(self, device: Optional[str] = None):
        self.model_name = settings.AUDIO_MODEL
        self.device = device or ("cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu")
        self._model = None
        self._feature_extractor = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info("Loading audio detector '%s' on %s...", self.model_name, self.device)
        started_at = time.perf_counter()
        model, feature_extractor = load_audio_classification_components(self.model_name)
        self._model = model.to(self.device)
        self._model.eval()
        self._feature_extractor = feature_extractor
        logger.info("Audio detector '%s' loaded in %.2fs", self.model_name, time.perf_counter() - started_at)

    def predict(self, audio: np.ndarray, sampling_rate: int) -> Dict[str, Any]:
        if self._model is None or self._feature_extractor is None:
            self._load_model()

        assert self._model is not None
        assert self._feature_extractor is not None

        started_at = time.perf_counter()
        try:
            inputs = self._feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            prob_real = float(probabilities[0].item())
            prob_fake = float(probabilities[1].item())

            return {
                "success": True,
                "detector": self.name,
                "model_used": self.model_name,
                "ai_probability": prob_fake,
                "human_probability": prob_real,
                "confidence": max(prob_real, prob_fake),
                "inference_time_ms": (time.perf_counter() - started_at) * 1000.0,
            }
        except Exception as exc:
            logger.error("Audio prediction failed for %s: %s", self.model_name, exc)
            return {
                "success": False,
                "detector": self.name,
                "model_used": self.model_name,
                "error": str(exc),
            }
