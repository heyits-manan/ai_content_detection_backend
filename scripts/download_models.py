from app.config import settings
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
    for key in settings.IMAGE_MODELS:
        model_name = IMAGE_MODELS.get(key)
        if not model_name:
            print(f"Skipping unsupported image model key: {key}")
            continue
        download_image_model(model_name)

    for key in settings.TEXT_MODELS:
        model_name = TEXT_MODELS.get(key)
        if not model_name:
            print(f"Skipping unsupported text model key: {key}")
            continue
        download_text_model(model_name)

    download_audio_model(settings.AUDIO_MODEL)

    print("All configured models downloaded and cached.")


if __name__ == "__main__":
    main()
