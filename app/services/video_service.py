"""
Video detection service.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Any, Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from fastapi import UploadFile

from app.config import settings
from app.core.exceptions import (
    BadRequestError,
    InferenceFailedError,
    UnprocessableEntityError,
)
from app.core.file_handler import remove_file_if_exists, resolve_upload_suffix, save_upload_to_temp
from app.models.image.base import clamp01

logger = logging.getLogger(__name__)

FrameSample = Tuple[int, np.ndarray]
FrameInferenceFn = Callable[[np.ndarray], float]
CONTENT_TYPE_SUFFIX_MAP = {
    "video/mp4": ".mp4",
    "application/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
}


class VideoValidationError(Exception):
    """Raised when an uploaded video is invalid."""


class VideoDecodeError(Exception):
    """Raised when the video cannot be decoded."""


class NoFramesExtractedError(Exception):
    """Raised when no usable frames can be extracted."""


class FrameInferenceError(Exception):
    """Raised when every frame inference fails."""


class FrameInferenceTimeoutError(Exception):
    """Raised when frame inference exceeds the configured timeout."""


def _safe_frame_count(capture: cv2.VideoCapture) -> int:
    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames is None or not math.isfinite(total_frames) or total_frames <= 0:
        return 0
    return max(0, int(total_frames))


def _count_decodable_frames(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return 0

    count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is not None and frame.size > 0:
                count += 1
    finally:
        capture.release()

    return count


def _compute_sample_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        return []

    sample_count = min(total_frames, max(1, num_frames))
    indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int).tolist()

    unique_indices: List[int] = []
    seen: set[int] = set()
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        unique_indices.append(index)
    return unique_indices


def _read_frame_with_retry(
    capture: cv2.VideoCapture,
    target_index: int,
    total_frames: int,
    max_forward_seek: int = 3,
) -> FrameSample | None:
    for offset in range(max_forward_seek + 1):
        candidate_index = target_index + offset
        if candidate_index >= total_frames:
            break

        capture.set(cv2.CAP_PROP_POS_FRAMES, candidate_index)
        success, frame = capture.read()
        if success and frame is not None and frame.size > 0:
            return candidate_index, frame

    return None


def extract_frames(video_path: str, num_frames: int) -> List[FrameSample]:
    """
    Extract uniformly sampled frames from a video without loading the full file.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than zero")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise VideoDecodeError("Failed to decode uploaded video")

    try:
        total_frames = _safe_frame_count(capture)
        if total_frames <= 0:
            capture.release()
            total_frames = _count_decodable_frames(video_path)
            if total_frames <= 0:
                raise NoFramesExtractedError("No frames extracted from video")
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                raise VideoDecodeError("Failed to reopen video for frame extraction")

        sample_indices = _compute_sample_indices(total_frames, num_frames)
        frames: List[FrameSample] = []
        seen_indices: set[int] = set()

        for index in sample_indices:
            sampled = _read_frame_with_retry(capture, index, total_frames)
            if sampled is None:
                logger.warning("Skipping undecodable frame near index %s for %s", index, video_path)
                continue

            sampled_index, frame = sampled
            if sampled_index in seen_indices:
                continue

            seen_indices.add(sampled_index)
            frames.append((sampled_index, frame))

        if not frames:
            raise NoFramesExtractedError("No frames extracted from video")

        return frames
    finally:
        capture.release()


def _run_frame_inference(frame_index: int, frame: np.ndarray, inference_fn: FrameInferenceFn) -> Dict[str, float | int]:
    probability = clamp01(float(inference_fn(frame)))
    return {"frame_index": frame_index, "probability": probability}


def _run_image_ensemble_inference(frame: np.ndarray) -> float:
    from app.services.image_service import run_image_ensemble

    return float(run_image_ensemble(frame))


def _ensure_image_ensemble_loaded() -> None:
    from app.services.image_service import ensure_image_ensemble_loaded

    ensure_image_ensemble_loaded()


def process_frames_parallel(
    frames: Sequence[FrameSample],
    max_workers: int,
    inference_timeout_seconds: float,
    inference_fn: FrameInferenceFn = _run_image_ensemble_inference,
) -> List[Dict[str, float | int]]:
    """
    Run frame inference concurrently and return sorted successful results.
    """
    if not frames:
        raise NoFramesExtractedError("No frames available for inference")

    worker_count = max(1, min(max_workers, len(frames)))
    batches = math.ceil(len(frames) / worker_count)
    timeout_budget_seconds = max(
        inference_timeout_seconds,
        float(batches * 45),
    )
    executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="video-infer")

    try:
        future_to_index = {
            executor.submit(_run_frame_inference, frame_index, frame, inference_fn): frame_index
            for frame_index, frame in frames
        }

        results: List[Dict[str, float | int]] = []
        started_at = time.perf_counter()
        logger.info(
            "Starting parallel frame inference: frames=%s workers=%s timeout_budget_seconds=%.2f",
            len(frames),
            worker_count,
            timeout_budget_seconds,
        )
        for future in as_completed(future_to_index, timeout=timeout_budget_seconds):
            frame_index = future_to_index[future]
            try:
                results.append(future.result())
                logger.info(
                    "Frame inference completed: frame_index=%s completed=%s/%s elapsed_seconds=%.2f",
                    frame_index,
                    len(results),
                    len(frames),
                    time.perf_counter() - started_at,
                )
            except Exception as exc:
                logger.warning("Frame inference failed for frame %s: %s", frame_index, exc)

        if not results:
            raise FrameInferenceError("Inference failed for all extracted frames")

        results.sort(key=lambda item: int(item["frame_index"]))
        return results
    except FuturesTimeoutError as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        raise FrameInferenceTimeoutError(
            f"Video frame inference timed out after {timeout_budget_seconds:.0f}s"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=False)


def aggregate_scores(frame_results: Sequence[Dict[str, float | int]]) -> float:
    """
    Combine frame scores with the required max/mean hybrid strategy.
    """
    if not frame_results:
        raise ValueError("frame_results must not be empty")

    probabilities = [float(frame_result["probability"]) for frame_result in frame_results]
    mean_probability = sum(probabilities) / len(probabilities)
    max_probability = max(probabilities)
    return clamp01((0.7 * max_probability) + (0.3 * mean_probability))


class VideoDetectionService:
    def __init__(
        self,
        *,
        default_num_frames: int | None = None,
        max_workers: int | None = None,
        inference_timeout_seconds: float | None = None,
    ) -> None:
        self.default_num_frames = default_num_frames or settings.VIDEO_DEFAULT_NUM_FRAMES
        self.max_workers = max_workers or settings.VIDEO_MAX_WORKERS
        self.inference_timeout_seconds = (
            inference_timeout_seconds or settings.VIDEO_INFERENCE_TIMEOUT_SECONDS
        )

    async def detect_from_upload(self, file: UploadFile, num_frames: int | None = None) -> Dict[str, Any]:
        temp_path: str | None = None
        started_at = time.perf_counter()

        try:
            temp_path = await self.save_validated_upload(file)
            requested_frames = num_frames or self.default_num_frames
            logger.info(
                "Starting video detection: filename=%s temp_path=%s requested_frames=%s workers=%s timeout_seconds=%s",
                file.filename,
                temp_path,
                requested_frames,
                self.max_workers,
                self.inference_timeout_seconds,
            )
            result = await asyncio.to_thread(self.detect_from_file, temp_path, requested_frames)
            result["processing_time_ms"] = round((time.perf_counter() - started_at) * 1000.0, 2)
            return result
        except VideoValidationError as exc:
            logger.warning("Video validation failed for filename=%s: %s", file.filename, exc)
            raise BadRequestError(str(exc)) from exc
        except VideoDecodeError as exc:
            logger.warning("Video decode failed for filename=%s: %s", file.filename, exc)
            raise UnprocessableEntityError(str(exc)) from exc
        except NoFramesExtractedError as exc:
            logger.warning("No frames extracted for filename=%s: %s", file.filename, exc)
            raise UnprocessableEntityError(str(exc)) from exc
        except FrameInferenceTimeoutError as exc:
            logger.warning("Video inference timed out for filename=%s: %s", file.filename, exc)
            raise InferenceFailedError(
                str(exc),
                details={"kind": "timeout"},
                status_code=504,
            ) from exc
        except FrameInferenceError as exc:
            logger.warning("Video inference failed for filename=%s: %s", file.filename, exc)
            raise InferenceFailedError(str(exc)) from exc
        except Exception:
            logger.exception("Unhandled video detection error for filename=%s", file.filename)
            raise
        finally:
            await file.close()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info("Removed temp video file: %s", temp_path)

    def detect_from_file(self, video_path: str, num_frames: int) -> Dict[str, Any]:
        overall_started_at = time.perf_counter()

        warmup_started_at = time.perf_counter()
        _ensure_image_ensemble_loaded()
        logger.info(
            "Image ensemble warmup completed in %.2f ms before video frame inference",
            (time.perf_counter() - warmup_started_at) * 1000.0,
        )

        extraction_started_at = time.perf_counter()
        frames = extract_frames(video_path, num_frames)
        extraction_time_ms = (time.perf_counter() - extraction_started_at) * 1000.0

        inference_started_at = time.perf_counter()
        frame_results = process_frames_parallel(
            frames=frames,
            max_workers=self.max_workers,
            inference_timeout_seconds=self.inference_timeout_seconds,
        )
        inference_time_ms = (time.perf_counter() - inference_started_at) * 1000.0

        final_score = aggregate_scores(frame_results)
        total_time_ms = (time.perf_counter() - overall_started_at) * 1000.0

        logger.info(
            "Video detection complete for %s: requested_frames=%s extracted_frames=%s used_frames=%s "
            "extraction_ms=%.2f inference_ms=%.2f total_ms=%.2f",
            video_path,
            num_frames,
            len(frames),
            len(frame_results),
            extraction_time_ms,
            inference_time_ms,
            total_time_ms,
        )

        return {
            "ai_probability": final_score,
            "frame_results": frame_results,
            "aggregation": "max_mean_hybrid",
            "num_frames_used": len(frame_results),
        }

    async def save_validated_upload(self, file: UploadFile) -> str:
        suffix = resolve_upload_suffix(
            filename=file.filename,
            content_type=file.content_type,
            allowed_extensions=settings.ALLOWED_VIDEO_EXTENSIONS,
            allowed_content_types=settings.ALLOWED_VIDEO_CONTENT_TYPES,
            content_type_suffix_map=CONTENT_TYPE_SUFFIX_MAP,
            generic_prefix="video/",
            default_suffix=".mp4",
        )
        if not suffix:
            if file.filename or file.content_type:
                raise VideoValidationError(
                    "Invalid video upload. "
                    f"filename={file.filename or '<missing>'}, content_type={(file.content_type or '<missing>').lower()}, "
                    f"allowed_extensions={settings.ALLOWED_VIDEO_EXTENSIONS}"
                )
            raise VideoValidationError("Invalid video upload. Missing filename extension and content type.")
        try:
            saved_upload = await save_upload_to_temp(
                file=file,
                upload_dir=settings.UPLOAD_DIR,
                prefix="video_",
                suffix=suffix,
                chunk_size=settings.VIDEO_UPLOAD_CHUNK_SIZE,
                max_size_bytes=settings.MAX_VIDEO_UPLOAD_SIZE,
            )
        except ValueError as exc:
            message = str(exc).replace("Uploaded file is empty", "Uploaded video file is empty")
            raise VideoValidationError(message) from exc

        logger.info(
            "Saved uploaded video to temp file: filename=%s temp_path=%s bytes=%s",
            file.filename,
            saved_upload.temp_path,
            saved_upload.size_bytes,
        )
        return saved_upload.temp_path
