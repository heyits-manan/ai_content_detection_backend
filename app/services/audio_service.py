from __future__ import annotations

import asyncio
import logging
import math
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import aiofiles
import librosa
import numpy as np
from fastapi import HTTPException, UploadFile

from app.config import settings
from app.models.audio.voice_detector import VoiceDeepfakeDetector
from app.models.image.base import clamp01

logger = logging.getLogger(__name__)


class AudioValidationError(Exception):
    """Raised when an uploaded audio file is invalid."""


class AudioDecodeError(Exception):
    """Raised when audio decoding fails."""


@lru_cache(maxsize=1)
def get_audio_detector() -> VoiceDeepfakeDetector:
    return VoiceDeepfakeDetector()


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
    min_chunk_seconds: float,
) -> List[Dict[str, Any]]:
    if audio.size == 0:
        return []

    chunk_size = max(1, int(chunk_seconds * sample_rate))
    hop_size = max(1, int((chunk_seconds - overlap_seconds) * sample_rate))
    min_chunk_size = max(1, int(min_chunk_seconds * sample_rate))

    if audio.shape[0] <= chunk_size:
        return [
            {
                "chunk_index": 0,
                "start_seconds": 0.0,
                "end_seconds": round(audio.shape[0] / sample_rate, 4),
                "samples": audio,
            }
        ]

    chunks: List[Dict[str, Any]] = []
    chunk_index = 0
    for start in range(0, audio.shape[0], hop_size):
        end = min(audio.shape[0], start + chunk_size)
        samples = audio[start:end]
        if samples.shape[0] < min_chunk_size:
            if not chunks:
                chunks.append(
                    {
                        "chunk_index": 0,
                        "start_seconds": 0.0,
                        "end_seconds": round(samples.shape[0] / sample_rate, 4),
                        "samples": samples,
                    }
                )
            break

        chunks.append(
            {
                "chunk_index": chunk_index,
                "start_seconds": round(start / sample_rate, 4),
                "end_seconds": round(end / sample_rate, 4),
                "samples": samples,
            }
        )
        chunk_index += 1

        if end >= audio.shape[0]:
            break

    return chunks


def aggregate_audio_scores(chunk_results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not chunk_results:
        raise ValueError("chunk_results must not be empty")

    probabilities = [float(chunk["probability"]) for chunk in chunk_results]
    mean_probability = sum(probabilities) / len(probabilities)
    max_probability = max(probabilities)
    ai_probability = clamp01((0.7 * max_probability) + (0.3 * mean_probability))

    return {
        "ai_probability": ai_probability,
        "human_probability": clamp01(1.0 - ai_probability),
        "average_ai_probability": clamp01(mean_probability),
        "max_ai_probability": clamp01(max_probability),
    }


class AudioDetectionService:
    def __init__(self) -> None:
        self.detector = get_audio_detector()

    async def detect_from_upload(self, file: UploadFile) -> Dict[str, Any]:
        temp_path: str | None = None
        started_at = time.perf_counter()

        try:
            suffix = self._validate_upload(file)
            temp_path = await self._save_upload_to_temp(file, suffix)
            result = await asyncio.to_thread(self.detect_from_file, temp_path, file.filename or "audio")
            result["processing_time_ms"] = round((time.perf_counter() - started_at) * 1000.0, 2)
            return result
        except AudioValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AudioDecodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            await file.close()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def detect_from_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        started_at = time.perf_counter()
        try:
            audio, sample_rate = librosa.load(
                file_path,
                sr=settings.AUDIO_SAMPLE_RATE,
                mono=True,
            )
        except Exception as exc:
            raise AudioDecodeError(f"Failed to decode audio file: {exc}") from exc

        if audio.size == 0:
            raise AudioDecodeError("Decoded audio file is empty")

        duration_seconds = round(audio.shape[0] / sample_rate, 4)
        chunks = chunk_audio(
            audio=audio,
            sample_rate=sample_rate,
            chunk_seconds=settings.AUDIO_CHUNK_SECONDS,
            overlap_seconds=settings.AUDIO_CHUNK_OVERLAP_SECONDS,
            min_chunk_seconds=settings.AUDIO_MIN_CHUNK_SECONDS,
        )
        if not chunks:
            raise AudioDecodeError("No audio chunks could be prepared for inference")

        chunk_results: List[Dict[str, Any]] = []
        for chunk in chunks:
            prediction = self.detector.predict(chunk["samples"], sample_rate)
            if prediction.get("success") is not True:
                logger.warning("Audio chunk inference failed for %s chunk=%s: %s", filename, chunk["chunk_index"], prediction.get("error"))
                continue

            chunk_results.append(
                {
                    "chunk_index": chunk["chunk_index"],
                    "start_seconds": chunk["start_seconds"],
                    "end_seconds": chunk["end_seconds"],
                    "probability": clamp01(float(prediction["ai_probability"])),
                }
            )

        if not chunk_results:
            raise HTTPException(status_code=500, detail="Audio inference failed for all chunks")

        aggregated = aggregate_audio_scores(chunk_results)
        processing_time_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info(
            "Audio detection complete for %s: chunks=%s used_chunks=%s duration_seconds=%.2f total_ms=%.2f",
            filename,
            len(chunks),
            len(chunk_results),
            duration_seconds,
            processing_time_ms,
        )

        return {
            "success": True,
            "ai_probability": aggregated["ai_probability"],
            "human_probability": aggregated["human_probability"],
            "average_ai_probability": aggregated["average_ai_probability"],
            "max_ai_probability": aggregated["max_ai_probability"],
            "is_ai_generated": aggregated["ai_probability"] > 0.5,
            "confidence": max(aggregated["ai_probability"], aggregated["human_probability"]),
            "aggregation": "max_mean_hybrid",
            "num_chunks_used": len(chunk_results),
            "duration_seconds": duration_seconds,
            "filename": filename,
            "chunk_results": chunk_results,
            "model_used": self.detector.model_name,
        }

    def _validate_upload(self, file: UploadFile) -> str:
        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()
        content_type = (file.content_type or "").lower()

        if suffix in settings.ALLOWED_AUDIO_EXTENSIONS:
            return suffix
        if content_type in settings.ALLOWED_AUDIO_CONTENT_TYPES or content_type.startswith("audio/"):
            return suffix or ".wav"
        raise AudioValidationError(
            f"Invalid audio upload. filename={filename or '<missing>'}, "
            f"content_type={content_type or '<missing>'}, "
            f"allowed_extensions={settings.ALLOWED_AUDIO_EXTENSIONS}"
        )

    async def _save_upload_to_temp(self, file: UploadFile, suffix: str) -> str:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            prefix="audio_",
            dir=settings.UPLOAD_DIR,
        ) as temp_file:
            temp_path = temp_file.name

        total_bytes = 0
        try:
            await file.seek(0)
            async with aiofiles.open(temp_path, "wb") as output_file:
                while True:
                    chunk = await file.read(settings.AUDIO_UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > settings.MAX_AUDIO_UPLOAD_SIZE:
                        raise AudioValidationError(
                            f"File too large. Max size: {settings.MAX_AUDIO_UPLOAD_SIZE / (1024 * 1024)}MB"
                        )
                    await output_file.write(chunk)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        if total_bytes == 0:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise AudioValidationError("Uploaded audio file is empty")

        return temp_path
