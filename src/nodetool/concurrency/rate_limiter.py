import asyncio
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiting behavior.

    Args:
        max_rate: Maximum number of operations allowed per time period.
        time_period: Time period in seconds for the rate limit.
        burst: Maximum number of operations allowed in a single burst (default: max_rate).

    Example:
        # Allow 10 operations per second with bursts up to 15
        config = RateLimitConfig(max_rate=10, time_period=1.0, burst=15)

        # Allow 60 operations per minute
        config = RateLimitConfig(max_rate=60, time_period=60.0)
    """

    max_rate: int
    time_period: float = 1.0
    burst: int | None = None

    def __post_init__(self):
        if self.max_rate <= 0:
            raise ValueError("max_rate must be a positive integer")
        if self.time_period <= 0:
            raise ValueError("time_period must be a positive number")
        if self.burst is not None and self.burst <= 0:
            raise ValueError("burst must be a positive integer")


class AsyncRateLimiter:
    """
    An async rate limiter using the token bucket algorithm.

    This provides time-based rate limiting to control the execution rate of
    async operations, useful for API rate limiting and preventing overload.

    Features:
    - Token bucket algorithm for smooth rate limiting
    - Burst support for handling temporary load spikes
    - Both context manager and decorator interfaces
    - Configurable rate limits

    Example:
        # Create a rate limiter: 10 operations per second, burst up to 15
        limiter = AsyncRateLimiter(
            max_rate=10,
            time_period=1.0,
            burst=15
        )

        # Using as context manager
        async with limiter:
            await make_api_request()

        # Using as decorator
        @limiter
        async def make_api_request():
            ...
    """

    def __init__(
        self,
        max_rate: int,
        time_period: float = 1.0,
        burst: int | None = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            max_rate: Maximum number of operations allowed per time period.
            time_period: Time period in seconds for the rate limit (default: 1.0).
            burst: Maximum burst size (default: same as max_rate).

        Raises:
            ValueError: If invalid configuration parameters are provided.
        """
        self.config = RateLimitConfig(
            max_rate=max_rate,
            time_period=time_period,
            burst=burst,
        )
        self._tokens = float(self.burst)
        self._last_update: float | None = None
        self._lock = asyncio.Lock()

    @property
    def max_rate(self) -> int:
        """Return the maximum rate of operations per time period."""
        return self.config.max_rate

    @property
    def time_period(self) -> float:
        """Return the time period in seconds."""
        return self.config.time_period

    @property
    def burst(self) -> int:
        """Return the maximum burst size."""
        return self.config.burst if self.config.burst is not None else self.config.max_rate

    async def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = asyncio.get_event_loop().time()

        if self._last_update is None:
            self._last_update = now
            return

        elapsed = now - self._last_update

        if elapsed >= self.config.time_period:
            self._tokens = float(self.burst)
            self._last_update = now
            return

        tokens_to_add = (elapsed / self.config.time_period) * self.config.max_rate
        self._tokens = min(self._tokens + tokens_to_add, float(self.burst))
        self._last_update = now

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the rate limiter.

        Args:
            tokens: Number of tokens to acquire (default: 1).

        Returns:
            Wait time in seconds before the operation can proceed.

        Raises:
            ValueError: If requesting more tokens than the burst limit.
        """
        if tokens > self.burst:
            raise ValueError(f"Cannot acquire {tokens} tokens at once (burst limit: {self.burst})")

        async with self._lock:
            await self._refill_tokens()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            tokens_needed = tokens - self._tokens
            time_needed = (tokens_needed / self.config.max_rate) * self.config.time_period

            self._tokens = 0
            self._last_update = asyncio.get_event_loop().time()

            await asyncio.sleep(time_needed)
            return time_needed

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Use as a decorator for rate-limiting async functions.

        Example:
            @AsyncRateLimiter(max_rate=10, time_period=1.0)
            async def make_api_request(url):
                ...
        """
        import functools

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self:
                return await func(*args, **kwargs)

        return wrapper


async def rate_limited(
    max_rate: int,
    time_period: float = 1.0,
    burst: int | None = None,
) -> AsyncRateLimiter:
    """
    Create a rate limiter for use as a context manager.

    Args:
        max_rate: Maximum number of operations per time period.
        time_period: Time period in seconds (default: 1.0).
        burst: Maximum burst size (default: same as max_rate).

    Returns:
        Configured AsyncRateLimiter instance.

    Example:
        async with await rate_limited(max_rate=10, time_period=1.0) as limiter:
            await make_api_request()
    """
    return AsyncRateLimiter(max_rate=max_rate, time_period=time_period, burst=burst)


__all__ = [
    "AsyncRateLimiter",
    "RateLimitConfig",
    "rate_limited",
]
