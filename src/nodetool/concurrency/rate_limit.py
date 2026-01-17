import asyncio
import time
from typing import TypeVar

T = TypeVar("T")


class AsyncTokenBucket:
    """
    An asynchronous token bucket rate limiter.

    The token bucket algorithm allows for controlled burstiness while maintaining
    a long-term rate limit. Tokens are added to the bucket at a fixed rate up to
    the bucket capacity. Each operation consumes one or more tokens.

    Example:
        # Allow 10 requests per second with burst capacity of 20
        limiter = AsyncTokenBucket(rate=10.0, capacity=20)

        # Acquire a token (blocks if no tokens available)
        async with limiter:
            await make_api_request()

        # Try to acquire without blocking
        if await limiter.try_acquire():
            await make_api_request()
    """

    def __init__(
        self,
        rate: float,
        capacity: int,
        initial_tokens: float | None = None,
    ):
        """
        Initialize the token bucket rate limiter.

        Args:
            rate: Rate at which tokens are added per second.
            capacity: Maximum number of tokens the bucket can hold.
            initial_tokens: Initial number of tokens (default: capacity).

        Raises:
            ValueError: If rate <= 0, capacity <= 0, or initial_tokens > capacity.
        """
        if rate <= 0:
            raise ValueError("rate must be a positive number")
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if initial_tokens is not None and initial_tokens > capacity:
            raise ValueError("initial_tokens cannot exceed capacity")

        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity if initial_tokens is None else initial_tokens
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def rate(self) -> float:
        """Return the rate at which tokens are added per second."""
        return self._rate

    @property
    def capacity(self) -> int:
        """Return the maximum number of tokens the bucket can hold."""
        return self._capacity

    @property
    def available_tokens(self) -> float:
        """Return the current number of available tokens."""
        return self._tokens

    def _add_tokens(self) -> None:
        """Add tokens based on elapsed time since last update."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """
        Acquire tokens from the bucket with optional timeout.

        Args:
            tokens: Number of tokens to acquire (default: 1).
            timeout: Maximum time to wait in seconds (default: wait indefinitely).

        Returns:
            True if tokens were acquired, False if timeout expired.

        Example:
            limiter = AsyncTokenBucket(rate=10, capacity=20)

            # Acquire 1 token, wait up to 5 seconds
            if await limiter.acquire(timeout=5.0):
                await make_api_request()
        """
        if tokens <= 0:
            raise ValueError("tokens must be a positive integer")
        if tokens > self._capacity:
            raise ValueError(f"Cannot acquire more than capacity ({self._capacity})")

        async with self._lock:
            self._add_tokens()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            if timeout is None:
                await self._wait_for_tokens(tokens)
                return True

            if timeout <= 0:
                return False

            return await self._wait_for_tokens_with_timeout(tokens, timeout)

    async def _wait_for_tokens(self, tokens: int) -> None:
        """Wait until enough tokens are available."""
        while self._tokens < tokens:
            deficit = tokens - self._tokens
            wait_time = deficit / self._rate
            await asyncio.sleep(wait_time)
            self._add_tokens()

    async def _wait_for_tokens_with_timeout(self, tokens: int, timeout: float) -> bool:
        """Wait for tokens with a timeout, returns True if successful."""
        start_time = time.monotonic()

        while self._tokens < tokens:
            elapsed = time.monotonic() - start_time
            remaining = timeout - elapsed

            if remaining <= 0:
                return False

            deficit = tokens - self._tokens
            wait_time = min(deficit / self._rate, remaining)
            await asyncio.sleep(wait_time)
            self._add_tokens()

        self._tokens -= tokens
        return True

    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire (default: 1).

        Returns:
            True if tokens were acquired, False otherwise.

        Example:
            limiter = AsyncTokenBucket(rate=10, capacity=20)

            if await limiter.try_acquire():
                await make_api_request()
        """
        if tokens <= 0:
            raise ValueError("tokens must be a positive integer")

        async with self._lock:
            self._add_tokens()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def __aenter__(self) -> "AsyncTokenBucket":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class AsyncRateLimiter:
    """
    A configurable async rate limiter using the token bucket algorithm.

    This class provides a high-level interface for rate limiting with sensible
    defaults and configuration options suitable for API rate limiting.

    Example:
        # 10 requests per second with burst of 20
        limiter = AsyncRateLimiter(rate=10, capacity=20)

        async with limiter:
            await make_api_request()

        # Custom rate limiting for different API tiers
        limiter = AsyncRateLimiter(
            rate=5,          # 5 requests per second
            capacity=10,     # Burst up to 10
            initial_tokens=5 # Start with 5 tokens
        )
    """

    def __init__(
        self,
        rate: float,
        capacity: int | None = None,
        initial_tokens: float | None = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            rate: Maximum rate of operations per second.
            capacity: Maximum burst capacity (defaults to rate if None).
            initial_tokens: Initial tokens (defaults to capacity if None).

        Raises:
            ValueError: If rate <= 0.
        """
        if rate <= 0:
            raise ValueError("rate must be a positive number")

        effective_capacity = capacity if capacity is not None else int(rate)
        effective_initial = initial_tokens if initial_tokens is not None else effective_capacity

        self._bucket = AsyncTokenBucket(
            rate=rate,
            capacity=effective_capacity,
            initial_tokens=effective_initial,
        )

    @property
    def rate(self) -> float:
        """Return the rate limit (operations per second)."""
        return self._bucket.rate

    @property
    def capacity(self) -> int:
        """Return the burst capacity."""
        return self._bucket.capacity

    @property
    def available_tokens(self) -> float:
        """Return the current number of available tokens."""
        return self._bucket.available_tokens

    async def acquire(self, timeout: float | None = None) -> bool:
        """
        Acquire permission to proceed with an operation.

        Args:
            timeout: Maximum time to wait in seconds (default: wait indefinitely).

        Returns:
            True if operation is allowed, False if timeout expired.

        Example:
            limiter = AsyncRateLimiter(rate=10, capacity=20)

            if await limiter.acquire(timeout=5.0):
                await make_api_request()
        """
        return await self._bucket.acquire(timeout=timeout)

    async def try_acquire(self) -> bool:
        """
        Try to acquire permission without blocking.

        Returns:
            True if operation is allowed, False otherwise.

        Example:
            limiter = AsyncRateLimiter(rate=10, capacity=20)

            if await limiter.try_acquire():
                await make_api_request()
        """
        return await self._bucket.try_acquire()

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self._bucket.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


__all__ = ["AsyncRateLimiter", "AsyncTokenBucket"]
