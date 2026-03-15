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


def _cache_only_error(model_name: str, exc: Exception) -> RuntimeError:
    error = RuntimeError(
        f"Model '{model_name}' is not available in the local Hugging Face cache at "
        f"'{settings.HF_HOME}'. Pre-download it during build or run scripts/download_models.py "
        f"with HF_HOME pointed at the same cache directory. Original error: {exc}"
    )
    error.__cause__ = exc
    return error


def load_image_pipeline(model_name: str, device: Optional[str | int] = None):
    from transformers import (  # type: ignore
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModelForImageClassification,
        pipeline,
    )

    common_kwargs = _hf_common_kwargs()
    try:
        model = AutoModelForImageClassification.from_pretrained(model_name, **common_kwargs)
    except Exception as exc:
        raise _cache_only_error(model_name, exc)

    pipeline_kwargs: Dict[str, Any] = {"model": model}
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, **common_kwargs)
        pipeline_kwargs["image_processor"] = processor
    except Exception:
        try:
            processor = AutoFeatureExtractor.from_pretrained(model_name, **common_kwargs)
            pipeline_kwargs["feature_extractor"] = processor
        except Exception as exc:
            raise _cache_only_error(model_name, exc)

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
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **common_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **common_kwargs)
    except Exception as exc:
        raise _cache_only_error(model_name, exc)

    pipeline_kwargs: Dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
    }
    if device is not None:
        pipeline_kwargs["device"] = device
    return pipeline("text-classification", **pipeline_kwargs)
