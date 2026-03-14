"""
Helpers for loading Hugging Face models strictly from the local cache when configured.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.config import settings


def _hf_common_kwargs() -> Dict[str, Any]:
    return {
        "cache_dir": settings.HF_HOME,
        "local_files_only": settings.HF_LOCAL_FILES_ONLY,
    }


def load_image_pipeline(model_name: str, device: Optional[str | int] = None):
    from transformers import (  # type: ignore
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModelForImageClassification,
        pipeline,
    )

    common_kwargs = _hf_common_kwargs()
    model = AutoModelForImageClassification.from_pretrained(model_name, **common_kwargs)

    pipeline_kwargs: Dict[str, Any] = {"model": model}
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, **common_kwargs)
        pipeline_kwargs["image_processor"] = processor
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(model_name, **common_kwargs)
        pipeline_kwargs["feature_extractor"] = processor

    if device is not None:
        pipeline_kwargs["device"] = device
    return pipeline("image-classification", **pipeline_kwargs)


def load_text_pipeline(model_name: str, device: Optional[str | int] = None):
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    common_kwargs = _hf_common_kwargs()
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **common_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **common_kwargs)

    pipeline_kwargs: Dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
    }
    if device is not None:
        pipeline_kwargs["device"] = device
    return pipeline("text-classification", **pipeline_kwargs)
