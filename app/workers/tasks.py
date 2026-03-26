from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

from app.config import settings
from app.core.file_handler import remove_file_if_exists


def _update_current_job_meta(status: str) -> None:
    try:
        from rq import get_current_job
    except ImportError:
        return

    job = get_current_job()
    if job is None:
        return

    job.meta["status"] = status
    job.save_meta()


def _build_mock_frame_results(num_frames: int) -> list[Dict[str, float | int]]:
    frame_count = max(1, min(num_frames, settings.VIDEO_DEFAULT_NUM_FRAMES))
    return [
        {
            "frame_index": index,
            "probability": round(min(0.95, 0.18 + (index * 0.03)), 4),
        }
        for index in range(frame_count)
    ]


def _build_mock_video_result(temp_path: str, num_frames: int) -> Dict[str, Any]:
    frame_results = _build_mock_frame_results(num_frames)
    average_probability = sum(float(item["probability"]) for item in frame_results) / len(frame_results)

    return {
        "ai_probability": round(average_probability, 4),
        "frame_results": frame_results,
        "aggregation": "mock_mean",
        "num_frames_used": len(frame_results),
        "processing_time_ms": round(settings.MOCK_VIDEO_PROCESSING_SECONDS * 1000.0, 2),
        "filename": Path(temp_path).name,
        "mode": "mock",
    }


def process_video_job(temp_path: str, num_frames: int) -> Dict[str, Any]:
    _update_current_job_meta("processing")

    try:
        if not settings.ENABLE_REAL_INFERENCE:
            time.sleep(settings.MOCK_VIDEO_PROCESSING_SECONDS)
            result = _build_mock_video_result(temp_path, num_frames)
        else:
            from app.services.video_service import VideoDetectionService

            service = VideoDetectionService()
            result = service.detect_from_file(temp_path, num_frames)

        _update_current_job_meta("completed")
        return result
    except Exception:
        _update_current_job_meta("failed")
        raise
    finally:
        if temp_path and os.path.exists(temp_path):
            remove_file_if_exists(temp_path)
