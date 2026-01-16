import asyncio
import time

import pytest

from nodetool.concurrency.rate_limiter import RateLimiter, rate_limited_gather


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid rate and burst."""
        limiter = RateLimiter(rate=10, burst=5)
        assert limiter.rate == 10
        assert limiter.burst == 5
        assert limiter.available_tokens == 5

    def test_init_with_default_burst(self):
        """Test that burst defaults to rate when not specified."""
        limiter = RateLimiter(rate=10)
        assert limiter.burst == 10
        assert limiter.available_tokens == 10

    def test_init_with_invalid_rate(self):
        """Test that invalid rate raises ValueError."""
        with pytest.raises(ValueError, match="rate must be a positive number"):
            RateLimiter(rate=0)

        with pytest.raises(ValueError, match="rate must be a positive number"):
            RateLimiter(rate=-1)

    def test_init_with_invalid_burst(self):
        """Test that invalid burst raises ValueError."""
        with pytest.raises(ValueError, match="burst must be a positive number"):
            RateLimiter(rate=10, burst=0)

        with pytest.raises(ValueError, match="burst must be a positive number"):
            RateLimiter(rate=10, burst=-1)

    def test_available_tokens_property(self):
        """Test available_tokens property returns correct value."""
        limiter = RateLimiter(rate=10, burst=5)
        assert limiter.available_tokens == 5

    @pytest.mark.asyncio
    async def test_acquire_returns_true_when_available(self):
        """Test acquire returns True when tokens are available."""
        limiter = RateLimiter(rate=10, burst=5)

        result = await limiter.acquire()
        assert result is True
        assert limiter.available_tokens == 4

    @pytest.mark.asyncio
    async def test_acquire_returns_false_when_rate_limited(self):
        """Test acquire returns False when rate limited."""
        limiter = RateLimiter(rate=1, burst=1)

        await limiter.acquire()
        assert limiter.available_tokens == 0

        result = await limiter.acquire()
        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens at once."""
        limiter = RateLimiter(rate=10, burst=5)

        result = await limiter.acquire(tokens=3)
        assert result is True
        assert limiter.available_tokens == 2

    @pytest.mark.asyncio
    async def test_acquire_invalid_token_count(self):
        """Test that acquiring invalid token count raises ValueError."""
        limiter = RateLimiter(rate=10, burst=5)

        with pytest.raises(ValueError, match="tokens must be a positive number"):
            await limiter.acquire(tokens=0)

        with pytest.raises(ValueError, match="tokens must be a positive number"):
            await limiter.acquire(tokens=-1)

    @pytest.mark.asyncio
    async def test_acquire_exceeds_burst(self):
        """Test that acquiring more than burst raises ValueError."""
        limiter = RateLimiter(rate=10, burst=5)

        with pytest.raises(ValueError, match="Cannot acquire more than burst"):
            await limiter.acquire(tokens=10)

    @pytest.mark.asyncio
    async def test_tokens_replenish_over_time(self):
        """Test that tokens replenish over time."""
        limiter = RateLimiter(rate=10, burst=5)

        await limiter.acquire(tokens=5)
        assert limiter.available_tokens == 0

        await asyncio.sleep(0.2)

        await limiter._update_tokens()
        assert limiter.available_tokens == pytest.approx(2.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        limiter = RateLimiter(rate=10, burst=5)

        async with limiter:
            assert limiter.available_tokens == 4

        assert limiter.available_tokens == 4

    @pytest.mark.asyncio
    async def test_context_manager_exception(self):
        """Test context manager releases on exception."""
        limiter = RateLimiter(rate=10, burst=5)

        with pytest.raises(ValueError):
            async with limiter:
                assert limiter.available_tokens == 4
                raise ValueError("test error")

        assert limiter.available_tokens == 4

    def test_reset(self):
        """Test reset restores full capacity."""
        limiter = RateLimiter(rate=10, burst=5)

        asyncio.run(limiter.acquire())
        assert limiter.available_tokens == 4

        limiter.reset()
        assert limiter.available_tokens == 5

    @pytest.mark.asyncio
    async def test_acquire_or_wait(self):
        """Test acquire_or_wait blocks until tokens available."""
        limiter = RateLimiter(rate=10, burst=1)

        await limiter.acquire()
        assert limiter.available_tokens == 0

        start = time.time()
        await limiter.acquire_or_wait()
        elapsed = time.time() - start

        assert elapsed >= 0.09
        assert limiter.available_tokens == 0

    @pytest.mark.asyncio
    async def test_acquire_or_wait_multiple_tokens(self):
        """Test acquire_or_wait with multiple tokens."""
        limiter = RateLimiter(rate=10, burst=5)

        await limiter.acquire_or_wait(tokens=5)
        assert limiter.available_tokens == 0

        start = time.time()
        await limiter.acquire_or_wait(tokens=3)
        elapsed = time.time() - start

        assert elapsed >= 0.29

    @pytest.mark.asyncio
    async def test_burst_capacity(self):
        """Test that burst capacity is respected."""
        limiter = RateLimiter(rate=1, burst=3)

        results = [await limiter.acquire() for _ in range(3)]
        assert results == [True, True, True]
        assert limiter.available_tokens < 0.01  # Allow small floating point error

    @pytest.mark.asyncio
    async def test_rate_enforcement(self):
        """Test that rate is enforced over time."""
        limiter = RateLimiter(rate=10, burst=10)

        start = time.time()
        for _ in range(10):
            await limiter.acquire()
        acquire_time = time.time() - start

        assert acquire_time < 0.1

        await limiter.acquire_or_wait()  # Consume the last token
        assert limiter.available_tokens < 0.01

        start = time.time()
        await limiter.acquire_or_wait()  # Need to wait for token to replenish
        wait_time = time.time() - start

        assert wait_time >= 0.09  # ~0.1s for 1 token at rate=10


class TestRateLimitedGather:
    """Tests for rate_limited_gather function."""

    @pytest.mark.asyncio
    async def test_empty_list(self):
        """Test with empty list returns empty list."""
        result = await rate_limited_gather([], rate=10, burst=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_rate(self):
        """Test that invalid rate raises ValueError."""
        with pytest.raises(ValueError, match="rate must be a positive number"):
            await rate_limited_gather([asyncio.sleep(0)], rate=0, burst=5)

    @pytest.mark.asyncio
    async def test_executes_all_tasks(self):
        """Test that all tasks are executed."""
        results = []

        async def task(n):
            await asyncio.sleep(0.01)
            results.append(n)
            return n * 2

        coros = [task(i) for i in range(3)]
        output = await rate_limited_gather(coros, rate=100, burst=10)

        assert output == [0, 2, 4]
        assert results == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        """Test that results are returned in the same order as input."""

        async def task(n):
            await asyncio.sleep(0.1 - n * 0.02)
            return n

        coros = [task(i) for i in range(5)]
        output = await rate_limited_gather(coros, rate=100, burst=10)

        assert output == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_respects_rate_limit(self):
        """Test that execution respects rate limit."""
        start_times = []
        lock = asyncio.Lock()

        async def task(n):
            async with lock:
                start_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)
            return n

        coros = [task(i) for i in range(10)]
        await rate_limited_gather(coros, rate=10, burst=5)

        total_duration = start_times[-1] - start_times[0]
        assert total_duration >= 0.4  # 5 tasks after burst: 0.1s between each at rate=10

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Test that exceptions are propagated correctly."""

        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("test error")

        coros = [asyncio.sleep(0.01), failing_task()]

        with pytest.raises(ValueError, match="test error"):
            await rate_limited_gather(coros, rate=100, burst=10)
