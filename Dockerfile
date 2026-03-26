FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/huggingface/hub /app/uploads

ENV HF_HOME=/opt/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY app app
COPY scripts scripts
RUN python scripts/download_models.py

ENV HF_LOCAL_FILES_ONLY=true \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    REDIS_URL=redis://redis:6379/0 \
    RQ_WORKER_CLASS=worker \
    ENABLE_REAL_INFERENCE=true \
    UPLOAD_DIR=/app/uploads

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
