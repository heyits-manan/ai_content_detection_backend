# AI Content Detection Backend

FastAPI backend for AI-generated content detection.

Current capabilities:
- Image detection with multiple Hugging Face models
- Text detection with multiple Hugging Face models
- Rate limiting with `slowapi`
- Docker build flow that pre-caches models for deployment

This README is written for someone cloning the backend and running it locally.

## Features

- `POST /api/v1/image/detect` for single image detection
- `POST /api/v1/image/detect-batch` for batch image detection
- `GET /api/v1/image/health` for image-service health
- `POST /api/v1/text/detect` for text detection
- `GET /api/v1/text/health` for text-service health
- `GET /health` for lightweight app health

## Models

Image models:
- `prithivMLmods/deepfake-detector-model-v1`
- `Organika/sdxl-detector`
- `umm-maybe/AI-image-detector`

Text models:
- `openai-community/roberta-base-openai-detector`
- `Hello-SimpleAI/chatgpt-detector-roberta`

## Requirements

- Python 3.13
- A virtual environment
- Internet access before first run to download the required models

## Local Setup

Choose the setup path that matches the client machine.

### macOS

#### 1. Clone and enter the backend

```bash
git clone <your-repo-url>
cd ai-content-detection/backend
```

#### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Download all models once

```bash
./venv/bin/python scripts/download_models.py
```

This downloads:
- `Organika/sdxl-detector`
- `prithivMLmods/deepfake-detector-model-v1`
- `umm-maybe/AI-image-detector`
- `openai-community/roberta-base-openai-detector`
- `Hello-SimpleAI/chatgpt-detector-roberta`

#### 5. Start the backend

```bash
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### 6. Open the API

- Root: `http://localhost:8000/`
- App health: `http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`

### Windows

Windows users have two practical options:
- `WSL2` with Ubuntu: recommended if they want a local Python setup similar to macOS/Linux
- `Docker Desktop`: recommended if they want the most reliable setup with the fewest Python package issues

#### Option A: Windows with WSL2

Inside the WSL Ubuntu shell:

```bash
git clone <your-repo-url>
cd ai-content-detection/backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
./venv/bin/python scripts/download_models.py
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Option B: Windows with Docker Desktop

From the backend folder:

```bash
docker build -t ai-content-backend .
docker run -p 8000:8000 ai-content-backend
```

Then open:
- Root: `http://localhost:8000/`
- App health: `http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`

### Local Runtime Behavior

- models are cached under `./.hf-cache` for non-Docker local runs
- runtime loads models from the local cache only
- `/health` does not load models
- model-loading happens lazily on the first request that needs them

## Local Model Cache

By default, local runs use:

```env
HF_HOME=./.hf-cache
HF_LOCAL_FILES_ONLY=true
```

This means:
- local development expects the models to already exist in `./.hf-cache`
- the downloaded files stay in `./.hf-cache`
- you should not commit that folder

If a model is missing, the backend will fail with a cache-related error instead of downloading it during a request.

## Environment Variables

The app reads settings from environment variables and `.env`.

Common variables:

```env
PROJECT_NAME="AI Content Detection ML Server"
USE_GPU=False

IMAGE_MODELS=["deepfake_v1","sdxl","umm_maybe"]
IMAGE_MODEL_WEIGHTS={"deepfake_v1":1.0,"sdxl":1.0,"umm_maybe":1.0}

TEXT_MODELS=["openai_roberta","hello_simpleai_roberta"]
TEXT_MODEL_WEIGHTS={"openai_roberta":1.0,"hello_simpleai_roberta":1.0}
```

Important notes:
- `IMAGE_MODELS` uses internal keys, not Hugging Face repo names
- valid image keys are `deepfake_v1`, `sdxl`, `umm_maybe`
- valid text keys are `openai_roberta`, `hello_simpleai_roberta`

## API Examples

### Image detection

```bash
curl -X POST "http://localhost:8000/api/v1/image/detect" \
  -F "file=@/absolute/path/to/image.png"
```

### Batch image detection

```bash
curl -X POST "http://localhost:8000/api/v1/image/detect-batch" \
  -F "files=@/absolute/path/to/image1.png" \
  -F "files=@/absolute/path/to/image2.png"
```

### Text detection

```bash
curl -X POST "http://localhost:8000/api/v1/text/detect" \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a sample paragraph to analyze."}'
```

### Lightweight health check

```bash
curl "http://localhost:8000/health"
```

### Service health

```bash
curl "http://localhost:8000/api/v1/image/health"
curl "http://localhost:8000/api/v1/text/health"
```

## Rate Limits

Current defaults:
- App default: `30/minute`
- Image detect: `10/minute`
- Image batch: `3/minute`
- Image health: `60/minute`
- Text detect: `20/minute`
- Text health: `60/minute`

## Docker and Render

This project is set up so Docker builds download models during image build, not at runtime.

Production container behavior:
- `HF_HOME=/opt/huggingface`
- models are downloaded during `docker build`
- runtime uses offline/cache-only mode
- the server starts without trying to download models on boot

### Local Docker build

```bash
docker build -t ai-content-backend .
docker run -p 8000:8000 ai-content-backend
```

### Render notes

Use a Docker-based Render service, not a plain Python service.

Why:
- the `Dockerfile` preloads the Hugging Face models into the container image
- runtime then loads only from the local container cache
- this avoids slow startup-time model downloads in production

## Troubleshooting

### Models are downloading locally on first request

That should not happen in the current setup. If it does, check whether `HF_LOCAL_FILES_ONLY` was overridden.

### Local run fails with cache or permission errors

Make sure you:
- are running from the backend directory so `./.hf-cache` resolves correctly
- ran `./venv/bin/python scripts/download_models.py` before starting the server

### Docker/Render runtime should not download models

If production logs show Hugging Face download requests:
- confirm Render is using the `Dockerfile`

## GitHub Actions CD To EC2 With Docker Hub

This repo now includes GitHub Actions workflows for:
- CI on pull requests and pushes to `main`
- building and pushing the Docker image to Docker Hub
- deploying the pushed image to an EC2 host over SSH

The Docker image keeps the existing container cache path:
- `HF_HOME=/opt/huggingface/hub`

Do not change that if you want Docker builds and runtime behavior to stay aligned.

### Required GitHub secrets

Add these repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `EC2_HOST`
- `EC2_USER`
- `EC2_SSH_KEY`
- `EC2_PORT` optional, defaults to `22`

### EC2 preparation

On the EC2 host:

1. Install Docker
2. Install `curl`
3. Create `/opt/ai-content-detection/.env`
4. Make sure the security group allows inbound traffic to your public app port

You can use:

```bash
chmod +x scripts/ec2-first-time-setup.sh
./scripts/ec2-first-time-setup.sh
```

### Deployment behavior

On every push to `main`, GitHub Actions will:

1. build the Docker image
2. push it to Docker Hub with `latest` and a SHA tag
3. SSH into EC2
4. pull the SHA-tagged image
5. replace the running container
6. call `/health` on the instance

The deployed container uses:
- container name: `ai-content-backend`
- host port: `8000`
- env file: `/opt/ai-content-detection/.env`

If you want a different image name, container name, or port, update `.github/workflows/deploy.yml`.
- confirm the build completed successfully
- confirm the service is Docker-based, not a plain Python web service

### Text or image request is slow on first use

That is usually model initialization, not startup failure.
First request latency is expected when the model is first loaded into memory.

## Project Structure

```text
backend/
├── app/
│   ├── api/
│   ├── models/
│   ├── services/
│   ├── core/
│   └── main.py
├── scripts/
│   └── download_models.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Notes

- `.hf-cache/` and `uploads/` are intentionally ignored by git
- model files should never be committed to the repository
- future video/audio detectors can follow the same Docker cache pattern
