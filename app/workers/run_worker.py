from __future__ import annotations

from app.config import settings
from app.core.redis_client import get_redis_connection


def main() -> None:
    try:
        from rq import Worker
    except ImportError as exc:
        raise RuntimeError(
            "RQ support is not installed. Add the 'rq' package before starting the worker."
        ) from exc

    connection = get_redis_connection()
    worker = Worker([settings.VIDEO_QUEUE_NAME], connection=connection)
    worker.work()


if __name__ == "__main__":
    main()
