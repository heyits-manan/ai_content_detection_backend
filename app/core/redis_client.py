from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import settings


@lru_cache(maxsize=1)
def get_redis_connection() -> Any:
    try:
        from redis import Redis
    except ImportError as exc:
        raise RuntimeError(
            "Redis support is not installed. Add the 'redis' package before using background jobs."
        ) from exc

    return Redis.from_url(settings.REDIS_URL)
