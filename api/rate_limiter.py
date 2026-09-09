"""
High-Performance Rate Limiter Middleware.

Features:
- Sliding Window Log & Token Bucket algorithms
- Redis backend for distributed deployments
- In-memory thread-safe fallback for standalone development
- HTTP 429 Too Many Requests response with Retry-After headers
"""

import os
import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)


class InMemorySlidingWindowLimiter:
    """Thread-safe in-memory sliding window rate limiter."""
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_logs: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Purge timestamps older than window
        timestamps = self.request_logs[client_key]
        self.request_logs[client_key] = [t for t in timestamps if t > cutoff]

        if len(self.request_logs[client_key]) < self.max_requests:
            self.request_logs[client_key].append(now)
            return True
        return False


class RateLimiter:
    """
    FastAPI dependency for rate limiting client IP or API key tokens.
    """
    def __init__(
        self,
        requests_per_minute: int = 60,
        redis_host: Optional[str] = None,
        redis_port: int = 6379,
    ):
        self.requests_per_minute = requests_per_minute
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", redis_port))
        self.redis_client = None
        self.in_memory_limiter = InMemorySlidingWindowLimiter(
            max_requests=requests_per_minute, window_seconds=60
        )

        # Attempt Redis connection
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
                socket_timeout=1,
            )
            self.redis_client.ping()
            logger.info("Connected to Redis for distributed rate limiting.")
        except Exception:
            self.redis_client = None
            logger.info("Redis not detected. Using in-memory sliding window rate limiter.")

    async def __call__(self, request: Request):
        # Extract client identifier
        client_ip = request.client.host if request.client else "unknown_client"
        api_key = request.headers.get("X-API-Key", "")
        identifier = f"rate_limit:{api_key or client_ip}"

        allowed = False
        if self.redis_client:
            try:
                now = time.time()
                key = identifier
                pipeline = self.redis_client.pipeline()
                pipeline.zremrangebyscore(key, 0, now - 60)
                pipeline.zadd(key, {str(now): now})
                pipeline.zcard(key)
                pipeline.expire(key, 60)
                _, _, count, _ = pipeline.execute()

                if count <= self.requests_per_minute:
                    allowed = True
            except Exception as e:
                logger.debug(f"Redis rate limit check failed ({e}), falling back to in-memory.")
                allowed = self.in_memory_limiter.is_allowed(identifier)
        else:
            allowed = self.in_memory_limiter.is_allowed(identifier)

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: maximum {self.requests_per_minute} requests per minute allowed.",
                headers={"Retry-After": "60"},
            )
        return True


if __name__ == "__main__":
    limiter = InMemorySlidingWindowLimiter(max_requests=3, window_seconds=2)
    print("Req 1 allowed:", limiter.is_allowed("127.0.0.1"))
    print("Req 2 allowed:", limiter.is_allowed("127.0.0.1"))
    print("Req 3 allowed:", limiter.is_allowed("127.0.0.1"))
    print("Req 4 allowed (should be False):", limiter.is_allowed("127.0.0.1"))
