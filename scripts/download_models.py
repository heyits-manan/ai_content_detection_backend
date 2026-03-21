from __future__ import annotations

import json
import os

from transformers import (  # type: ignore
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

IMAGE_MODELS = {
    "sdxl": "Organika/sdxl-detector",
    "deepfake_v1": "prithivMLmods/deepfake-detector-model-v1",
}

TEXT_MODELS = {
    "openai_roberta": "openai-community/roberta-base-openai-detector",
    "hello_simpleai_roberta": "Hello-SimpleAI/chatgpt-detector-roberta",
}


def _parse_list_env(name: str, default: list[str]) -> list[str]:
    raw_value = os.getenv(name)
    if not raw_value:
        return default

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return default

    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    return default


def download_image_model(model_name: str) -> None:
    print(f"Downloading image model: {model_name}")
    AutoModelForImageClassification.from_pretrained(model_name)
    try:
        AutoImageProcessor.from_pretrained(model_name)
    except Exception:
        AutoFeatureExtractor.from_pretrained(model_name)


def download_text_model(model_name: str) -> None:
    print(f"Downloading text model: {model_name}")
    AutoModelForSequenceClassification.from_pretrained(model_name)
    AutoTokenizer.from_pretrained(model_name)


def download_audio_model(model_name: str) -> None:
    print(f"Downloading audio model: {model_name}")
    AutoModelForAudioClassification.from_pretrained(model_name)
    AutoFeatureExtractor.from_pretrained(model_name)


def main() -> None:
    image_model_keys = _parse_list_env("IMAGE_MODELS", ["deepfake_v1", "sdxl", "umm_maybe"])
    text_model_keys = _parse_list_env("TEXT_MODELS", ["openai_roberta", "hello_simpleai_roberta"])
    audio_model_name = os.getenv("AUDIO_MODEL", "garystafford/wav2vec2-deepfake-voice-detector")

    for key in image_model_keys:
        model_name = IMAGE_MODELS.get(key)
        if not model_name:
            print(f"Skipping unsupported image model key: {key}")
            continue
        download_image_model(model_name)

    for key in text_model_keys:
        model_name = TEXT_MODELS.get(key)
        if not model_name:
            print(f"Skipping unsupported text model key: {key}")
            continue
        download_text_model(model_name)

    download_audio_model(audio_model_name)

    print("All configured models downloaded and cached.")


if __name__ == "__main__":
    main()
