from transformers import (  # type: ignore
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

IMAGE_MODELS = [
    "Organika/sdxl-detector",
    "prithivMLmods/deepfake-detector-model-v1",
    "umm-maybe/AI-image-detector",
]

TEXT_MODELS = [
    "openai-community/roberta-base-openai-detector",
    "Hello-SimpleAI/chatgpt-detector-roberta",
]


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


def main() -> None:
    for model_name in IMAGE_MODELS:
        download_image_model(model_name)

    for model_name in TEXT_MODELS:
        download_text_model(model_name)

    print("All configured models downloaded and cached.")


if __name__ == "__main__":
    main()
