from transformers import pipeline

print("Downloading SDXL image model...")
pipeline("image-classification", model="Organika/sdxl-detector")

print("Downloading deepfake image model...")
pipeline("image-classification", model="prithivMLmods/deepfake-detector-model-v1")

print("Downloading third image model...")
pipeline("image-classification", model="umm-maybe/AI-image-detector")

print("All image models downloaded and cached!")
