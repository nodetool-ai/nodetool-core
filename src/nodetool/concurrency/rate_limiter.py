import asyncio
from dataclasses import dataclass
from typing import Any, Callable

T = type("T", (), {})


@dataclass
class RateLimitResult:
    """Result of a rate limiter acquisition attempt."""

    allowed: bool
    wait_time: float


class AsyncRateLimiter:
    """
    An async rate limiter using the token bucket algorithm.

    This provides rate limiting for controlling the frequency of operations,
    such as API calls to external services with rate limits.

    Args:
        rate: Number of tokens to add per interval (e.g., 10 requests).
        interval: Time period in seconds for token replenishment (e.g., 1.0 for per-second).
        initial_tokens: Number of tokens available at initialization (default: rate).

    Example:
        # Allow 10 requests per second
        limiter = AsyncRateLimiter(rate=10, interval=1.0)

        # Use as context manager (blocks until rate limit allows)
        async with limiter:
            await make_api_call()

        # Use acquire for non-blocking check
        result = await limiter.acquire()
        if result.allowed:
            await make_api_call()
    """

    def __init__(
        self,
        rate: float,
        interval: float = 1.0,
        initial_tokens: float | None = None,
    ):
        if rate <= 0:
            raise ValueError("rate must be a positive number")
        if interval <= 0:
            raise ValueError("interval must be a positive number")

        self.rate = rate
        self.interval = interval
        self.tokens_per_second = rate / interval
        self._capacity = rate

        initial = initial_tokens if initial_tokens is not None else rate
        self._tokens = float(initial)
        self._last_update: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> float:
        """Return the maximum token capacity."""
        return self._capacity

    @property
    def available_tokens(self) -> float:
        """Return the current number of available tokens."""
        try:
            now = asyncio.get_event_loop().time()
        except RuntimeError:
            return self._tokens
        self._recalculate_tokens(now)
        return self._tokens

    def _recalculate_tokens(self, now: float) -> None:
        """Recalculate available tokens based on elapsed time."""
        if self._last_update > 0:
            elapsed = now - self._last_update
            new_tokens = elapsed * self.tokens_per_second
            self._tokens = min(self._capacity, self._tokens + new_tokens)
        self._last_update = now

    async def acquire(self, tokens: float = 1.0) -> RateLimitResult:
        """
        Acquire tokens from the rate limiter.

        Args:
            tokens: Number of tokens to acquire (default: 1).

        Returns:
            RateLimitResult with allowed status and wait time if not allowed.
        """
        if tokens <= 0:
            raise ValueError("tokens must be a positive number")
        if tokens > self._capacity:
            raise ValueError(f"Cannot acquire more tokens ({tokens}) than capacity ({self._capacity})")

        async with self._lock:
            now = asyncio.get_event_loop().time()
            self._recalculate_tokens(now)

            if self._tokens >= tokens:
                self._tokens -= tokens
                return RateLimitResult(allowed=True, wait_time=0.0)

            needed = tokens - self._tokens
            wait_time = needed / self.tokens_per_second

            await asyncio.sleep(wait_time)

            self._tokens = self._capacity - tokens
            self._last_update = asyncio.get_event_loop().time()

            return RateLimitResult(allowed=True, wait_time=wait_time)

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


async def rate_limited(
    func: Callable[..., Any],
    rate_limiter: AsyncRateLimiter,
) -> Any:
    """
    Execute a function with rate limiting.

    This is a convenience function for wrapping single operations with rate limiting.

    Args:
        func: Async function to execute.
        rate_limiter: RateLimiter instance to use.

    Returns:
        The result of the function.

    Example:
        limiter = AsyncRateLimiter(rate=10, interval=1.0)
        result = await rate_limited(make_api_call, limiter)
    """
    async with rate_limiter:
        result = func()
        if asyncio.iscoroutine(result):
            return await result
        return result


__all__ = ["AsyncRateLimiter", "RateLimitResult", "rate_limited"]
