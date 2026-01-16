import asyncio
from typing import TypeVar

T = TypeVar("T")


class RateLimiter:
    """
    An async rate limiter using the token bucket algorithm.

    This provides a flexible rate limiting mechanism that allows for burst traffic
    while maintaining a steady average rate. Tokens are added at a fixed rate up
    to a maximum capacity.

    Example:
        limiter = RateLimiter(rate=10, burst=5)

        # Using async context manager
        async with limiter:
            await make_api_call()

        # Using acquire directly
        if await limiter.acquire():
            try:
                await make_api_call()
            finally:
                pass  # Release happens automatically
    """

    def __init__(
        self,
        rate: float,
        burst: int | None = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            rate: Rate of token replenishment per second.
            burst: Maximum number of tokens that can be accumulated (bucket capacity).
                   Defaults to the same value as rate.

        Raises:
            ValueError: If rate is not positive or burst is not positive.
        """
        if rate <= 0:
            raise ValueError("rate must be a positive number")

        if burst is not None and burst <= 0:
            raise ValueError("burst must be a positive number")

        self._rate = rate
        self._burst = burst if burst is not None else int(rate)
        self._tokens = float(self._burst)
        try:
            self._last_update = asyncio.get_event_loop().time()
        except RuntimeError:
            self._last_update = 0.0
        self._lock = asyncio.Lock()

    @property
    def rate(self) -> float:
        """Return the token replenishment rate per second."""
        return self._rate

    @property
    def burst(self) -> int:
        """Return the maximum bucket capacity."""
        return self._burst

    @property
    def available_tokens(self) -> float:
        """Return the current number of available tokens."""
        return self._tokens

    async def _update_tokens(self) -> None:
        """Update the number of tokens based on elapsed time."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_update

        if elapsed > 0:
            added = elapsed * self._rate
            self._tokens = min(self._tokens + added, self._burst)
            self._last_update = now

    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Acquire tokens from the rate limiter.

        Args:
            tokens: Number of tokens to acquire (default: 1).

        Returns:
            True if tokens were acquired, False if rate limited.
        """
        if tokens <= 0:
            raise ValueError("tokens must be a positive number")

        if tokens > self._burst:
            raise ValueError(f"Cannot acquire more than burst capacity ({self._burst})")

        async with self._lock:
            await self._update_tokens()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    async def acquire_or_wait(self, tokens: float = 1.0) -> None:
        """
        Acquire tokens, waiting if necessary to respect the rate limit.

        This method will block until the required tokens are available.

        Args:
            tokens: Number of tokens to acquire (default: 1).

        Raises:
            ValueError: If tokens is not positive or exceeds burst capacity.
        """
        if tokens <= 0:
            raise ValueError("tokens must be a positive number")

        if tokens > self._burst:
            raise ValueError(f"Cannot acquire more than burst capacity ({self._burst})")

        async with self._lock:
            await self._update_tokens()

            while self._tokens < tokens:
                wait_time = (tokens - self._tokens) / self._rate
                self._tokens = 0
                self._last_update = asyncio.get_event_loop().time()
                await asyncio.sleep(wait_time)
                await self._update_tokens()

            self._tokens -= tokens

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        self._tokens = float(self._burst)
        try:
            self._last_update = asyncio.get_event_loop().time()
        except RuntimeError:
            self._last_update = 0.0


async def rate_limited_gather(
    coro_list: list,
    rate: float,
    burst: int | None = None,
) -> list:
    """
    Run coroutines with rate limiting.

    This function wraps asyncio.gather to limit the rate of coroutine execution.
    Coroutines are executed in order, with rate limiting applied.

    Args:
        coro_list: List of coroutine functions or awaitables to execute.
        rate: Rate of token replenishment per second.
        burst: Maximum number of tokens that can be accumulated.

    Returns:
        List of results in the same order as the input coroutines.

    Raises:
        ValueError: If rate is not positive or burst is not positive.

    Example:
        async def fetch_url(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()

        urls = [...]  # list of URLs
        results = await rate_limited_gather(
            [fetch_url(url) for url in urls],
            rate=10,  # 10 requests per second
            burst=5,  # allow bursts up to 5
        )
    """
    limiter = RateLimiter(rate=rate, burst=burst)

    async def run_with_limit(index: int, coro):
        await limiter.acquire_or_wait()
        return await coro

    return await asyncio.gather(*(run_with_limit(i, coro) for i, coro in enumerate(coro_list)))


__all__ = ["RateLimiter", "rate_limited_gather"]
