from __future__ import annotations

from app.config import settings
from app.core.redis_client import get_redis_connection


def _resolve_worker_class():
    try:
        from rq import SimpleWorker, Worker
    except ImportError as exc:
        raise RuntimeError(
            "RQ support is not installed. Add the 'rq' package before starting the worker."
        ) from exc

    worker_class = settings.RQ_WORKER_CLASS.strip().lower()
    if worker_class == "simple":
        return SimpleWorker
    if worker_class == "worker":
        return Worker

    raise RuntimeError(
        "Invalid RQ_WORKER_CLASS setting. Use 'simple' for local development or 'worker' for production."
    )


def main() -> None:
    connection = get_redis_connection()
    worker_class = _resolve_worker_class()
    worker = worker_class([settings.VIDEO_QUEUE_NAME], connection=connection)
    worker.work()


if __name__ == "__main__":
    main()
