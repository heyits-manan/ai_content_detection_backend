from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Optional

from app.models.image.base import BaseImageDetector
from app.models.image.sdxl_detector import SDXLDetector
from app.models.image.deepfake_detector import DeepfakeDetectorV1
from app.models.image.umm_maybe_detector import UMMMaybeDetector

logger = logging.getLogger(__name__)


DetectorKey = str


def available_detector_keys() -> List[DetectorKey]:
    return sorted(_DETECTOR_FACTORIES.keys())


def create_detector(key: DetectorKey) -> BaseImageDetector:
    try:
        factory = _DETECTOR_FACTORIES[key]
    except KeyError as e:
        raise ValueError(f"Unknown detector '{key}'. Available: {available_detector_keys()}") from e
    return factory()


@lru_cache(maxsize=1)
def get_detectors(keys: tuple[DetectorKey, ...]) -> List[BaseImageDetector]:
    """
    Process-level cache of instantiated detectors.

    NOTE: This is important because `ImageDetectionService` is currently created per-request
    via FastAPI dependency injection.
    """
    detectors: List[BaseImageDetector] = []
    for key in keys:
        logger.info(f"Loading image detector '{key}'...")
        detectors.append(create_detector(key))
    return detectors


_DETECTOR_FACTORIES: Dict[DetectorKey, callable] = {
    "sdxl": lambda: SDXLDetector(),
    "deepfake_v1": lambda: DeepfakeDetectorV1(),
    "umm_maybe": lambda: UMMMaybeDetector(),
}
