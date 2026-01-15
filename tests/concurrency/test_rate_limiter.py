import asyncio
import time

import pytest

from nodetool.concurrency.rate_limiter import (
    AsyncRateLimiter,
    RateLimitConfig,
    rate_limited,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig class."""

    def test_default_values(self):
        """Test default RateLimitConfig values."""
        config = RateLimitConfig(max_rate=10)
        assert config.max_rate == 10
        assert config.time_period == 1.0
        assert config.burst is None

    def test_custom_values(self):
        """Test custom RateLimitConfig values."""
        config = RateLimitConfig(max_rate=60, time_period=60.0, burst=30)
        assert config.max_rate == 60
        assert config.time_period == 60.0
        assert config.burst == 30

    def test_burst_defaults_to_none(self):
        """Test that burst defaults to None in config."""
        config = RateLimitConfig(max_rate=10)
        assert config.burst is None

    def test_invalid_max_rate(self):
        """Test that invalid max_rate raises ValueError."""
        with pytest.raises(ValueError, match="max_rate must be a positive integer"):
            RateLimitConfig(max_rate=0)

        with pytest.raises(ValueError, match="max_rate must be a positive integer"):
            RateLimitConfig(max_rate=-1)

    def test_invalid_time_period(self):
        """Test that invalid time_period raises ValueError."""
        with pytest.raises(ValueError, match="time_period must be a positive number"):
            RateLimitConfig(max_rate=10, time_period=0)

        with pytest.raises(ValueError, match="time_period must be a positive number"):
            RateLimitConfig(max_rate=10, time_period=-1.0)

    def test_invalid_burst(self):
        """Test that invalid burst raises ValueError."""
        with pytest.raises(ValueError, match="burst must be a positive integer"):
            RateLimitConfig(max_rate=10, burst=0)

        with pytest.raises(ValueError, match="burst must be a positive integer"):
            RateLimitConfig(max_rate=10, burst=-1)


class TestAsyncRateLimiter:
    """Tests for AsyncRateLimiter class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        limiter = AsyncRateLimiter(max_rate=10)
        assert limiter.max_rate == 10
        assert limiter.time_period == 1.0
        assert limiter.burst == 10

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        limiter = AsyncRateLimiter(max_rate=60, time_period=60.0, burst=30)
        assert limiter.max_rate == 60
        assert limiter.time_period == 60.0
        assert limiter.burst == 30

    def test_invalid_max_rate(self):
        """Test that invalid max_rate raises ValueError."""
        with pytest.raises(ValueError, match="max_rate must be a positive integer"):
            AsyncRateLimiter(max_rate=0)

        with pytest.raises(ValueError, match="max_rate must be a positive integer"):
            AsyncRateLimiter(max_rate=-1)

    def test_invalid_burst(self):
        """Test that invalid burst raises ValueError."""
        with pytest.raises(ValueError, match="burst must be a positive integer"):
            AsyncRateLimiter(max_rate=10, burst=0)

        with pytest.raises(ValueError, match="burst must be a positive integer"):
            AsyncRateLimiter(max_rate=10, burst=-1)

    @pytest.mark.asyncio
    async def test_acquire_returns_zero_when_tokens_available(self):
        """Test that acquire returns 0 when tokens are available."""
        limiter = AsyncRateLimiter(max_rate=10, burst=10)
        wait_time = await limiter.acquire()
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens at once."""
        limiter = AsyncRateLimiter(max_rate=10, burst=10)
        wait_time = await limiter.acquire(tokens=5)
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_too_many_tokens(self):
        """Test that acquiring more tokens than burst raises ValueError."""
        limiter = AsyncRateLimiter(max_rate=10, burst=10)
        with pytest.raises(ValueError, match="Cannot acquire 15 tokens"):
            await limiter.acquire(tokens=15)

    @pytest.mark.asyncio
    async def test_rate_limiting_basic(self):
        """Test that rate limiting actually limits the rate."""
        limiter = AsyncRateLimiter(max_rate=10, time_period=1.0, burst=10)

        start = time.time()
        for _ in range(10):
            await limiter.acquire()
        elapsed = time.time() - start

        assert elapsed < 0.2  # All should complete quickly since burst = max_rate

    @pytest.mark.asyncio
    async def test_rate_limiting_enforces_delay(self):
        """Test that rate limiting enforces delay when burst is exceeded."""
        limiter = AsyncRateLimiter(max_rate=10, time_period=1.0, burst=5)

        start = time.time()
        for _ in range(7):
            await limiter.acquire()
        elapsed = time.time() - start

        assert elapsed >= 0.1  # At least 2 tokens worth of delay

    pass

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using rate limiter as context manager."""
        limiter = AsyncRateLimiter(max_rate=10, time_period=1.0, burst=10)

        async with limiter:
            pass  # Should acquire token successfully

    @pytest.mark.asyncio
    async def test_context_manager_allows_burst(self):
        """Test that context manager respects burst limit."""
        limiter = AsyncRateLimiter(max_rate=10, time_period=1.0, burst=5)

        start = time.time()
        for _ in range(5):
            async with limiter:
                pass
        elapsed = time.time() - start

        assert elapsed < 0.1  # All bursts should complete quickly

        wait_time = await limiter.acquire()
        assert wait_time > 0  # Next one should wait


class TestAsyncRateLimiterDecorator:
    """Tests for AsyncRateLimiter decorator usage."""

    @pytest.mark.asyncio
    async def test_decorator_limits_calls(self):
        """Test that decorator limits function call rate."""
        limiter = AsyncRateLimiter(max_rate=5, time_period=1.0, burst=5)

        call_count = 0

        @limiter
        async def my_function():
            nonlocal call_count
            call_count += 1
            return call_count

        start = time.time()
        results = []
        for _ in range(5):
            results.append(await my_function())
        elapsed = time.time() - start

        assert results == [1, 2, 3, 4, 5]
        assert elapsed < 0.1  # All bursts complete quickly

        next_result = await my_function()
        assert next_result == 6
        elapsed = time.time() - start
        assert elapsed >= 0.1  # Should have waited for rate limit

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""
        limiter = AsyncRateLimiter(max_rate=10)

        @limiter
        async def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    @pytest.mark.asyncio
    async def test_decorator_passes_arguments(self):
        """Test that decorator passes arguments to wrapped function."""
        limiter = AsyncRateLimiter(max_rate=10)

        @limiter
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(2, 3)
        assert result == 5


class TestRateLimitedFunction:
    """Tests for the rate_limited helper function."""

    @pytest.mark.asyncio
    async def test_rate_limited_creates_limiter(self):
        """Test that rate_limited creates a rate limiter."""
        limiter = await rate_limited(max_rate=10, time_period=1.0, burst=10)
        assert isinstance(limiter, AsyncRateLimiter)
        assert limiter.max_rate == 10
        assert limiter.burst == 10


class TestAsyncRateLimiterConcurrency:
    """Tests for concurrent access to the rate limiter."""

    @pytest.mark.asyncio
    async def test_concurrent_acquire_requests(self):
        """Test that concurrent acquire requests are handled correctly."""
        limiter = AsyncRateLimiter(max_rate=10, time_period=1.0, burst=10)

        async def acquire_and_record():
            wait_time = await limiter.acquire()
            return wait_time

        start = time.time()
        tasks = [acquire_and_record() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        assert all(r == 0.0 for r in results)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_concurrent_burst_across_time(self):
        """Test rate limiting with concurrent tasks over time."""
        limiter = AsyncRateLimiter(max_rate=10, time_period=1.0, burst=5)

        results = []

        async def do_work():
            async with limiter:
                results.append(time.time())
                await asyncio.sleep(0.01)

        await asyncio.gather(*[do_work() for _ in range(5)])
        first_duration = results[-1] - results[0]

        assert first_duration < 0.1  # All should complete in first burst

        await asyncio.gather(*[do_work() for _ in range(3)])
        second_duration = results[-1] - results[-3]

        assert second_duration >= 0.1  # Should have waited for rate limit
