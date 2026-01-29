import asyncio
import time

import pytest

from nodetool.concurrency.rate_limit import AsyncRateLimiter, AsyncTokenBucket


class TestAsyncTokenBucket:
    """Tests for AsyncTokenBucket class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=20)
        assert bucket.rate == 10.0
        assert bucket.capacity == 20
        assert bucket.available_tokens == 20

    def test_init_with_custom_initial_tokens(self):
        """Test initialization with custom initial tokens."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=20, initial_tokens=5)
        assert bucket.available_tokens == 5

    def test_init_with_invalid_rate(self):
        """Test that invalid rate raises ValueError."""
        with pytest.raises(ValueError, match="rate must be a positive number"):
            AsyncTokenBucket(rate=0, capacity=10)

        with pytest.raises(ValueError, match="rate must be a positive number"):
            AsyncTokenBucket(rate=-1, capacity=10)

    def test_init_with_invalid_capacity(self):
        """Test that invalid capacity raises ValueError."""
        with pytest.raises(ValueError, match="capacity must be a positive integer"):
            AsyncTokenBucket(rate=10, capacity=0)

        with pytest.raises(ValueError, match="capacity must be a positive integer"):
            AsyncTokenBucket(rate=10, capacity=-1)

    def test_init_with_excessive_initial_tokens(self):
        """Test that excessive initial_tokens raises ValueError."""
        with pytest.raises(ValueError, match="initial_tokens cannot exceed capacity"):
            AsyncTokenBucket(rate=10, capacity=5, initial_tokens=10)

    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        """Test basic acquire and release functionality."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=5)

        assert bucket.available_tokens == 5

        await bucket.acquire()
        assert bucket.available_tokens == 4

        await bucket.acquire()
        assert 3.0 <= bucket.available_tokens <= 4.0

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens at once."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=10)

        await bucket.acquire(tokens=3)
        assert bucket.available_tokens == 7

        await bucket.acquire(tokens=5)
        assert 1.9 <= bucket.available_tokens <= 2.1

    @pytest.mark.asyncio
    async def test_acquire_invalid_tokens(self):
        """Test that invalid token count raises ValueError."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=10)

        with pytest.raises(ValueError, match="tokens must be a positive integer"):
            await bucket.acquire(tokens=0)

        with pytest.raises(ValueError, match="tokens must be a positive integer"):
            await bucket.acquire(tokens=-1)

    @pytest.mark.asyncio
    async def test_acquire_exceeds_capacity(self):
        """Test that acquiring more than capacity raises ValueError."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=5)

        with pytest.raises(ValueError, match="Cannot acquire more than capacity"):
            await bucket.acquire(tokens=10)

    @pytest.mark.asyncio
    async def test_try_acquire_success(self):
        """Test successful non-blocking acquire."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=5)

        result = await bucket.try_acquire()
        assert result is True
        assert bucket.available_tokens == 4

    @pytest.mark.asyncio
    async def test_try_acquire_failure(self):
        """Test failed non-blocking acquire."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=2)

        await bucket.acquire(tokens=2)
        assert bucket.available_tokens < 1

        result = await bucket.try_acquire()
        assert result is False

    @pytest.mark.asyncio
    async def test_token_bucket_context_manager(self):
        """Test async context manager usage."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=5)

        async with bucket:
            assert bucket.available_tokens == 4

        assert bucket.available_tokens == 4

    @pytest.mark.asyncio
    async def test_token_bucket_context_manager_exception(self):
        """Test context manager handles exceptions correctly."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=5)
        initial_tokens = bucket.available_tokens

        with pytest.raises(ValueError):
            async with bucket:
                raise ValueError("test error")

        assert bucket.available_tokens < initial_tokens

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self):
        """Test that tokens are refilled over time."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=10, initial_tokens=1)

        await bucket.acquire()
        tokens_after_acquire = bucket.available_tokens

        await asyncio.sleep(0.2)

        bucket._add_tokens()
        assert bucket.available_tokens > tokens_after_acquire

    @pytest.mark.asyncio
    async def test_token_bucket_acquire_with_timeout_success(self):
        """Test acquire with timeout succeeds when tokens available."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=5)

        result = await bucket.acquire(timeout=1.0)
        assert result is True
        assert bucket.available_tokens == 4

    @pytest.mark.asyncio
    async def test_token_bucket_acquire_with_timeout_expires(self):
        """Test acquire with timeout when bucket is empty."""
        bucket = AsyncTokenBucket(rate=1.0, capacity=1)

        await bucket.acquire()
        assert bucket.available_tokens < 1

        start = time.time()
        result = await bucket.acquire(timeout=0.2)
        elapsed = time.time() - start

        assert result is False
        assert 0.15 <= elapsed < 0.5

    @pytest.mark.asyncio
    async def test_token_bucket_acquire_with_zero_timeout(self):
        """Test acquire with zero timeout returns immediately."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=1)

        result = await bucket.acquire(timeout=0)
        assert result is True

        result = await bucket.acquire(timeout=0)
        assert result is False

    @pytest.mark.asyncio
    async def test_token_bucket_caps_at_capacity(self):
        """Test that tokens don't exceed capacity."""
        bucket = AsyncTokenBucket(rate=10.0, capacity=5, initial_tokens=0)

        await asyncio.sleep(1.0)
        bucket._add_tokens()

        assert bucket.available_tokens == 5

    @pytest.mark.asyncio
    async def test_token_bucket_multiple_concurrent_acquires(self):
        """Test multiple tasks acquiring from the same bucket."""
        bucket = AsyncTokenBucket(rate=100.0, capacity=10)
        acquired_count = 0
        lock = asyncio.Lock()

        async def acquire_and_wait():
            nonlocal acquired_count
            async with bucket:
                async with lock:
                    acquired_count += 1
                await asyncio.sleep(0.05)

        tasks = [acquire_and_wait() for _ in range(5)]
        await asyncio.gather(*tasks)

        assert acquired_count == 5


class TestAsyncRateLimiter:
    """Tests for AsyncRateLimiter class."""

    def test_rate_limiter_init_with_rate_only(self):
        """Test initialization with rate only."""
        limiter = AsyncRateLimiter(rate=10)

        assert limiter.rate == 10
        assert limiter.capacity == 10
        assert limiter.available_tokens == 10

    def test_rate_limiter_init_with_rate_and_capacity(self):
        """Test initialization with rate and capacity."""
        limiter = AsyncRateLimiter(rate=5, capacity=20)

        assert limiter.rate == 5
        assert limiter.capacity == 20
        assert limiter.available_tokens == 20

    def test_rate_limiter_init_with_custom_initial_tokens(self):
        """Test initialization with custom initial tokens."""
        limiter = AsyncRateLimiter(rate=10, capacity=20, initial_tokens=5)

        assert limiter.available_tokens == 5

    def test_rate_limiter_init_with_invalid_rate(self):
        """Test that invalid rate raises ValueError."""
        with pytest.raises(ValueError, match="rate must be a positive number"):
            AsyncRateLimiter(rate=0)

        with pytest.raises(ValueError, match="rate must be a positive number"):
            AsyncRateLimiter(rate=-1)

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """Test basic acquire functionality."""
        limiter = AsyncRateLimiter(rate=10, capacity=20)

        result = await limiter.acquire()
        assert result is True
        assert limiter.available_tokens == 19

    @pytest.mark.asyncio
    async def test_rate_limiter_try_acquire(self):
        """Test try_acquire functionality."""
        limiter = AsyncRateLimiter(rate=2, capacity=2)

        await limiter.acquire()
        await limiter.acquire()
        assert limiter.available_tokens < 1

        result = await limiter.try_acquire()
        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limiter_context_manager(self):
        """Test async context manager usage."""
        limiter = AsyncRateLimiter(rate=10, capacity=20)

        async with limiter:
            assert limiter.available_tokens == 19

        assert limiter.available_tokens == 19

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_with_timeout(self):
        """Test acquire with timeout."""
        limiter = AsyncRateLimiter(rate=1, capacity=1)

        await limiter.acquire()
        assert limiter.available_tokens < 1

        result = await limiter.acquire(timeout=0.2)
        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limiter_rate_limiting_behavior(self):
        """Test that rate limiting is enforced over time."""
        limiter = AsyncRateLimiter(rate=10, capacity=10)

        start = time.time()
        for _ in range(10):
            await limiter.acquire()
        elapsed = time.time() - start

        assert elapsed < 0.5

        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        assert elapsed >= 0.09
