from __future__ import annotations

from typing import Any, Dict, Protocol

import numpy as np


class BaseAudioDetector(Protocol):
    name: str
    model_name: str

    def predict(self, audio: np.ndarray, sampling_rate: int) -> Dict[str, Any]:
        """
        Return a dict with audio classification probabilities.
        """
