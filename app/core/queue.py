from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import settings
from app.core.redis_client import get_redis_connection


@lru_cache(maxsize=1)
def get_video_queue() -> Any:
    try:
        from rq import Queue
    except ImportError as exc:
        raise RuntimeError(
            "RQ support is not installed. Add the 'rq' package before using background jobs."
        ) from exc

    return Queue(
        name=settings.VIDEO_QUEUE_NAME,
        connection=get_redis_connection(),
        default_timeout=settings.VIDEO_JOB_TIMEOUT_SECONDS,
    )


def fetch_job_by_id(job_id: str) -> Any:
    try:
        from rq.job import Job
    except ImportError as exc:
        raise RuntimeError(
            "RQ support is not installed. Add the 'rq' package before using background jobs."
        ) from exc

    return Job.fetch(job_id, connection=get_redis_connection())
