"""Per-caller request rate limiting.

Uploads run document parsing and, in production, an LLM extraction call, so an
unthrottled caller burns real money and can starve everyone else on the same
worker. The limiter is keyed by API key when one is present and by client IP
otherwise, so a shared deployment cannot have one tenant exhaust another's share.

Two backends, matching the rest of the project:

- "memory" is a per-process sliding window. Correct for a single worker, and
  honestly wrong for several — four uvicorn workers each admit the full quota, so
  the effective limit is four times what was configured.
- "redis" keeps the window in Redis, so every worker and every machine shares one
  counter. This is the one to use whenever more than one process serves traffic.
"""
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int


class RateLimiter(Protocol):
    def check(self, identity: str) -> RateLimitDecision: ...


class NullRateLimiter:
    """Used when the limit is zero, so callers need no special-casing."""

    def check(self, identity: str) -> RateLimitDecision:
        return RateLimitDecision(allowed=True, limit=0, remaining=0, retry_after_seconds=0)


class InMemorySlidingWindowLimiter:
    """Sliding window over request timestamps, scoped to this process."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, identity: str) -> RateLimitDecision:
        now = time.monotonic()
        hits = self._hits[identity]

        # A true sliding window, not a fixed bucket: a fixed bucket lets a caller
        # send the whole quota at 0:59 and the whole quota again at 1:00.
        cutoff = now - self._window
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self._limit:
            retry_after = max(1, int(self._window - (now - hits[0])) + 1)
            return RateLimitDecision(
                allowed=False, limit=self._limit, remaining=0, retry_after_seconds=retry_after
            )

        hits.append(now)
        return RateLimitDecision(
            allowed=True,
            limit=self._limit,
            remaining=self._limit - len(hits),
            retry_after_seconds=0,
        )


class RedisSlidingWindowLimiter:
    """Shared window in Redis, so every worker counts against one quota."""

    def __init__(self, limit: int, redis_url: str, window_seconds: int = 60) -> None:
        import redis

        self._limit = limit
        self._window = window_seconds
        self._client = redis.Redis.from_url(redis_url)

    def check(self, identity: str) -> RateLimitDecision:
        key = f"ratelimit:{identity}"

        # INCR then EXPIRE in one round trip; the first request of a window sets
        # the TTL, and the key disappears on its own when the window closes.
        pipeline = self._client.pipeline()
        pipeline.incr(key)
        pipeline.ttl(key)
        count, ttl = pipeline.execute()

        if ttl < 0:
            self._client.expire(key, self._window)
            ttl = self._window

        if count > self._limit:
            return RateLimitDecision(
                allowed=False, limit=self._limit, remaining=0, retry_after_seconds=max(1, int(ttl))
            )

        return RateLimitDecision(
            allowed=True,
            limit=self._limit,
            remaining=max(0, self._limit - int(count)),
            retry_after_seconds=0,
        )


@lru_cache
def get_rate_limiter() -> RateLimiter:
    settings = get_settings()

    if settings.rate_limit_per_minute <= 0:
        return NullRateLimiter()

    if settings.rate_limit_backend == "redis":
        return RedisSlidingWindowLimiter(settings.rate_limit_per_minute, settings.redis_url)

    return InMemorySlidingWindowLimiter(settings.rate_limit_per_minute)


def caller_identity(api_key: str | None, client_host: str | None) -> str:
    """API key first, client IP as the fallback.

    Keying on IP alone would put every user behind one corporate NAT into a single
    bucket; keying on the API key gives each tenant its own quota.
    """
    if api_key:
        return f"key:{api_key}"
    return f"ip:{client_host or 'unknown'}"
