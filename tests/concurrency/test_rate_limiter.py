import asyncio
import time

import pytest

from nodetool.concurrency.rate_limiter import AsyncRateLimiter, RateLimitResult, rate_limited


class TestAsyncRateLimiter:
    """Tests for AsyncRateLimiter class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        limiter = AsyncRateLimiter(rate=10, interval=1.0)
        assert limiter.rate == 10
        assert limiter.interval == 1.0
        assert limiter.capacity == 10
        assert 9.99 < limiter.available_tokens < 10.01

    def test_init_with_custom_initial_tokens(self):
        """Test initialization with custom initial tokens."""
        limiter = AsyncRateLimiter(rate=10, interval=1.0, initial_tokens=5)
        assert limiter.capacity == 10
        assert 4.99 < limiter.available_tokens < 5.01

    def test_init_with_invalid_rate(self):
        """Test that invalid rate raises ValueError."""
        with pytest.raises(ValueError, match="rate must be a positive number"):
            AsyncRateLimiter(rate=0, interval=1.0)

        with pytest.raises(ValueError, match="rate must be a positive number"):
            AsyncRateLimiter(rate=-1, interval=1.0)

    def test_init_with_invalid_interval(self):
        """Test that invalid interval raises ValueError."""
        with pytest.raises(ValueError, match="interval must be a positive number"):
            AsyncRateLimiter(rate=10, interval=0)

        with pytest.raises(ValueError, match="interval must be a positive number"):
            AsyncRateLimiter(rate=10, interval=-1)

    @pytest.mark.asyncio
    async def test_acquire_single_token(self):
        """Test acquiring a single token."""
        limiter = AsyncRateLimiter(rate=10, interval=1.0)

        result = await limiter.acquire()
        assert result.allowed is True
        assert result.wait_time == 0.0
        assert 8.99 < limiter.available_tokens < 9.01

    @pytest.mark.asyncio
    async def test_acquire_exceeds_capacity(self):
        """Test that acquiring more than capacity raises ValueError."""
        limiter = AsyncRateLimiter(rate=10, interval=1.0)

        with pytest.raises(ValueError, match="Cannot acquire more tokens"):
            await limiter.acquire(tokens=15)

    @pytest.mark.asyncio
    async def test_acquire_zero_tokens(self):
        """Test that acquiring zero tokens raises ValueError."""
        limiter = AsyncRateLimiter(rate=10, interval=1.0)

        with pytest.raises(ValueError, match="tokens must be a positive number"):
            await limiter.acquire(tokens=0)

    @pytest.mark.asyncio
    async def test_rate_limiting_enforces_rate(self):
        """Test that rate limiting properly enforces the rate."""
        limiter = AsyncRateLimiter(rate=2, interval=1.0)

        start = time.time()
        await limiter.acquire()
        await limiter.acquire()

        result = await limiter.acquire()
        elapsed = time.time() - start

        assert result.allowed is True
        assert elapsed >= 0.4

    @pytest.mark.asyncio
    async def test_token_replenishment(self):
        """Test that tokens are replenished over time."""
        limiter = AsyncRateLimiter(rate=10, interval=1.0)

        await limiter.acquire(tokens=10)
        assert limiter.available_tokens < 0.01

        await asyncio.sleep(0.5)

        assert 4.5 < limiter.available_tokens < 5.5

    @pytest.mark.asyncio
    async def test_multiple_concurrent_acquires(self):
        """Test multiple tasks acquiring from the same limiter."""
        limiter = AsyncRateLimiter(rate=10, interval=1.0)
        results: list[RateLimitResult] = []

        async def acquire_and_record():
            result = await limiter.acquire()
            results.append(result)
            await asyncio.sleep(0.1)

        tasks = [acquire_and_record() for _ in range(5)]
        await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.allowed for r in results)


class TestRateLimited:
    """Tests for rate_limited function."""

    @pytest.mark.asyncio
    async def test_rate_limited_execution(self):
        """Test rate_limited function executes the function."""
        limiter = AsyncRateLimiter(rate=10, interval=1.0)
        call_count = 0

        async def count_calls():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await rate_limited(count_calls, limiter)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limited_consumes_tokens(self):
        """Test that rate_limited consumes tokens."""
        limiter = AsyncRateLimiter(rate=5, interval=1.0)

        async def noop():
            pass

        await rate_limited(noop, limiter)
        await rate_limited(noop, limiter)

        assert 2.99 < limiter.available_tokens < 3.01

    @pytest.mark.asyncio
    async def test_rate_limited_with_sync_function(self):
        """Test rate_limited with a sync function."""
        limiter = AsyncRateLimiter(rate=10, interval=1.0)
        call_count = 0

        def sync_func():
            nonlocal call_count
            call_count += 1
            return "sync_result"

        result = await rate_limited(sync_func, limiter)
        assert result == "sync_result"
        assert call_count == 1


class TestRateLimitResult:
    """Tests for RateLimitResult dataclass."""

    def test_allowed_result(self):
        """Test RateLimitResult for allowed operation."""
        result = RateLimitResult(allowed=True, wait_time=0.0)
        assert result.allowed is True
        assert result.wait_time == 0.0

    def test_denied_result(self):
        """Test RateLimitResult for denied operation."""
        result = RateLimitResult(allowed=False, wait_time=0.5)
        assert result.allowed is False
        assert result.wait_time == 0.5
