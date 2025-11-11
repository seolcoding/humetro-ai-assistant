"""
Concurrency Limiter for Vertex AI API Calls
Prevents grpc BlockingIOError by limiting concurrent requests
"""

import asyncio
import functools
from typing import Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class VertexAIConcurrencyLimiter:
    """
    Limits concurrent Vertex AI API calls to prevent grpc resource exhaustion

    Issues fixed:
    - BlockingIOError: [Errno 35] Resource temporarily unavailable
    - Empty responses from async calls
    - grpc PollerCompletionQueue thread conflicts
    """

    def __init__(self, max_concurrent_requests: int = 3):
        """
        Initialize concurrency limiter

        Args:
            max_concurrent_requests: Maximum concurrent API calls (default: 3)
                - Recommended: 3-5 for Vertex AI
                - Too high: grpc resource exhaustion
                - Too low: slow evaluation
        """
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.max_concurrent = max_concurrent_requests
        logger.info(f"🔒 VertexAI Concurrency Limiter initialized: max={max_concurrent_requests}")

    async def __aenter__(self):
        """Acquire semaphore"""
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release semaphore"""
        self.semaphore.release()

    def limit_concurrency(self, func: Callable) -> Callable:
        """
        Decorator to limit function concurrency

        Usage:
            @limiter.limit_concurrency
            async def api_call():
                ...
        """
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with self.semaphore:
                return await func(*args, **kwargs)
        return wrapper


# Global limiter instance
_global_limiter: Optional[VertexAIConcurrencyLimiter] = None


def get_limiter(max_concurrent: int = 3) -> VertexAIConcurrencyLimiter:
    """Get or create global limiter"""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = VertexAIConcurrencyLimiter(max_concurrent)
    return _global_limiter


def reset_limiter():
    """Reset global limiter (for testing)"""
    global _global_limiter
    _global_limiter = None
