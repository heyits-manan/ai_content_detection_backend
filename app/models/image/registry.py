from __future__ import annotations

import logging
import threading
from typing import Dict, List

from app.models.image.base import BaseImageDetector
from app.models.image.sdxl_detector import SDXLDetector
from app.models.image.deepfake_detector import DeepfakeDetectorV1
from app.models.image.umm_maybe_detector import UMMMaybeDetector

logger = logging.getLogger(__name__)


DetectorKey = str
_DETECTOR_CACHE: Dict[tuple[DetectorKey, ...], List[BaseImageDetector]] = {}
_DETECTOR_CACHE_LOCK = threading.Lock()


def available_detector_keys() -> List[DetectorKey]:
    return sorted(_DETECTOR_FACTORIES.keys())


def create_detector(key: DetectorKey) -> BaseImageDetector:
    try:
        factory = _DETECTOR_FACTORIES[key]
    except KeyError as e:
        raise ValueError(f"Unknown detector '{key}'. Available: {available_detector_keys()}") from e
    return factory()


def get_detectors(keys: tuple[DetectorKey, ...]) -> List[BaseImageDetector]:
    """
    Process-level cache of instantiated detectors.

    NOTE: This is important because `ImageDetectionService` is currently created per-request
    via FastAPI dependency injection.
    """
    cached = _DETECTOR_CACHE.get(keys)
    if cached is not None:
        return cached

    with _DETECTOR_CACHE_LOCK:
        cached = _DETECTOR_CACHE.get(keys)
        if cached is not None:
            return cached

        detectors: List[BaseImageDetector] = []
        for key in keys:
            logger.info(f"Loading image detector '{key}'...")
            detectors.append(create_detector(key))

        _DETECTOR_CACHE[keys] = detectors
        return detectors


_DETECTOR_FACTORIES: Dict[DetectorKey, callable] = {
    "sdxl": lambda: SDXLDetector(),
    "deepfake_v1": lambda: DeepfakeDetectorV1(),
    "umm_maybe": lambda: UMMMaybeDetector(),
}
