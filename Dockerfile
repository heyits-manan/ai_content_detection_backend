FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    TRANSFORMERS_CACHE=/opt/huggingface \
    HUGGINGFACE_HUB_CACHE=/opt/huggingface/hub \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN mkdir -p /opt/huggingface /app/uploads

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py

ENV HF_LOCAL_FILES_ONLY=true \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app app
COPY scripts scripts

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
