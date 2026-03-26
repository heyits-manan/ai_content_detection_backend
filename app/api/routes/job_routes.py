from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.models.response_models import VideoJobAcceptedResponse, VideoJobStatusResponse
from app.config import settings
from app.core.file_handler import remove_file_if_exists
from app.core.queue import fetch_job_by_id, get_video_queue
from app.core.rate_limit import limiter
from app.services.video_service import VideoDetectionService
from app.workers.tasks import process_video_job

router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_video_service_singleton() -> VideoDetectionService:
    return VideoDetectionService()


async def get_video_service() -> VideoDetectionService:
    return _get_video_service_singleton()


def _map_job_status(job: Any) -> str:
    meta_status = job.meta.get("status")
    if meta_status in {"processing", "completed", "failed"}:
        if meta_status == "processing":
            return "processing"
        if meta_status == "completed":
            return "completed"
        return "failed"

    status = job.get_status(refresh=True)
    if status in {"queued", "deferred", "scheduled"}:
        return "queued"
    if status in {"started"}:
        return "processing"
    if status == "finished":
        return "completed"
    return "failed"


def _extract_job_error(job: Any) -> str | None:
    exc_info = getattr(job, "exc_info", None)
    if isinstance(exc_info, str) and exc_info.strip():
        return exc_info.strip().splitlines()[-1]
    return None


@router.post("/video/jobs", response_model=VideoJobAcceptedResponse)
@limiter.limit(settings.VIDEO_DETECT_RATE_LIMIT)
async def create_video_job(
    request: Request,
    file: UploadFile = File(..., description="Video file to analyze"),
    num_frames: int = Form(settings.VIDEO_DEFAULT_NUM_FRAMES, ge=1, le=256),
    service: VideoDetectionService = Depends(get_video_service),
):
    temp_path = await service.save_validated_upload(file)

    try:
        queue = get_video_queue()
        job = queue.enqueue(
            process_video_job,
            temp_path,
            num_frames,
            job_timeout=settings.VIDEO_JOB_TIMEOUT_SECONDS,
            result_ttl=settings.JOB_RESULT_TTL_SECONDS,
            meta={"status": "queued"},
        )
    except Exception:
        remove_file_if_exists(temp_path)
        logger.exception("Failed to enqueue video job for filename=%s", file.filename)
        raise
    finally:
        await file.close()

    logger.info(
        "Enqueued video job: job_id=%s filename=%s temp_path=%s num_frames=%s",
        job.id,
        file.filename,
        temp_path,
        num_frames,
    )
    return VideoJobAcceptedResponse(success=True, job_id=job.id, status="queued")


@router.get("/jobs/{job_id}", response_model=VideoJobStatusResponse)
@limiter.limit(settings.JOB_STATUS_RATE_LIMIT)
async def get_job_status(request: Request, job_id: str) -> VideoJobStatusResponse:
    try:
        job = fetch_job_by_id(job_id)
    except Exception as exc:
        not_found_types = ("NoSuchJobError", "InvalidJobOperation")
        if exc.__class__.__name__ in not_found_types:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        raise

    status = _map_job_status(job)
    response = VideoJobStatusResponse(success=True, job_id=job_id, status=status)
    if status == "completed" and job.result is not None:
        response.result = job.result
    elif status == "failed":
        response.error = _extract_job_error(job) or "Video job failed"
    return response
