import asyncio

import pytest

from nodetool.concurrency.retry import RetryPolicy, retry_with_exponential_backoff


class TestRetryWithExponentialBackoff:
    """Tests for retry_with_exponential_backoff function."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Test that successful operations return immediately."""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_with_exponential_backoff(success_func)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Test that retries occur on failure."""
        call_count = 0

        async def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return "success"

        result = await retry_with_exponential_backoff(
            fail_twice,
            max_retries=3,
            initial_delay=0.01,
        )
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Test that exception is raised after max_retries."""

        async def always_fail():
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            await retry_with_exponential_backoff(
                always_fail,
                max_retries=2,
                initial_delay=0.01,
            )

    @pytest.mark.asyncio
    async def test_respects_initial_delay(self):
        """Test that initial delay is applied between attempts."""
        start_times = []

        async def record_time():
            start_times.append(asyncio.get_running_loop().time())
            if len(start_times) < 3:
                raise ValueError("fail")
            return "success"

        await retry_with_exponential_backoff(
            record_time,
            max_retries=3,
            initial_delay=0.05,
            jitter=False,
        )

        delay1 = start_times[1] - start_times[0]
        delay2 = start_times[2] - start_times[1]

        assert delay1 >= 0.04
        assert delay2 >= 0.09  # Exponential backoff: 0.05 * 2 = 0.10

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test that delays increase exponentially."""
        delays = []

        async def record_delay():
            if len(delays) < 2:
                raise ValueError("fail")
            return "success"

        async def record_time():
            now = asyncio.get_running_loop().time()
            if not delays:
                delays.append(now)
            else:
                delays.append(now - delays[0])
            if len(delays) < 3:
                raise ValueError("fail")
            return "success"

        await retry_with_exponential_backoff(
            record_time,
            max_retries=3,
            initial_delay=0.05,
            max_delay=10.0,
            jitter=False,
        )

        total_time = delays[2]
        assert total_time >= 0.05 + 0.10  # initial + initial * 2

    @pytest.mark.asyncio
    async def test_jitter_prevents_thundering_herd(self):
        """Test that jitter adds randomness to delays."""
        delays = []

        async def record_time_and_fail():
            now = asyncio.get_running_loop().time()
            delays.append(now)
            if len(delays) < 3:
                raise ValueError("fail")
            return "success"

        await retry_with_exponential_backoff(
            record_time_and_fail,
            max_retries=3,
            initial_delay=0.1,
            jitter=True,
        )

        assert len(delays) == 3

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        delays = []

        async def record_time_and_fail():
            now = asyncio.get_running_loop().time()
            delays.append(now)
            if len(delays) < 5:
                raise ValueError("fail")
            return "success"

        await retry_with_exponential_backoff(
            record_time_and_fail,
            max_retries=10,
            initial_delay=1.0,
            max_delay=2.0,
            jitter=False,
        )

        assert len(delays) == 5

    @pytest.mark.asyncio
    async def test_filtered_exceptions(self):
        """Test that only specified exceptions are retried."""
        call_count = 0

        async def raise_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await retry_with_exponential_backoff(
                raise_type_error,
                max_retries=3,
                initial_delay=0.01,
                retryable_exceptions=(ValueError,),
            )

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_unlimited_retries(self):
        """Test that max_retries=-1 enables unlimited retries."""
        call_count = 0

        async def fail_many():
            nonlocal call_count
            call_count += 1
            if call_count < 10:
                raise ValueError("failing")
            return "success"

        result = await retry_with_exponential_backoff(
            fail_many,
            max_retries=-1,
            initial_delay=0.001,
            jitter=False,
        )

        assert result == "success"
        assert call_count == 10

    @pytest.mark.asyncio
    async def test_invalid_max_retries(self):
        """Test that invalid max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be -1"):
            await retry_with_exponential_backoff(
                lambda: asyncio.sleep(0),
                max_retries=-2,
            )

    @pytest.mark.asyncio
    async def test_lambda_function(self):
        """Test that lambda functions work correctly."""
        call_count = 0

        async def make_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("fail")
            return call_count * 10

        result = await retry_with_exponential_backoff(
            make_func,
            max_retries=3,
            initial_delay=0.01,
        )

        assert result == 20


class TestRetryPolicy:
    """Tests for RetryPolicy class."""

    def test_default_initialization(self):
        """Test default RetryPolicy values."""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.initial_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.exponential_base == 2.0
        assert policy.jitter is True

    @pytest.mark.asyncio
    async def test_execute_method(self):
        """Test RetryPolicy.execute method."""
        policy = RetryPolicy(max_retries=2, initial_delay=0.01)
        call_count = 0

        async def succeed_on_third():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"

        result = await policy.execute(succeed_on_third)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_custom_retryable_exceptions(self):
        """Test RetryPolicy with custom retryable exceptions."""
        policy = RetryPolicy(
            max_retries=2,
            initial_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        call_count = 0

        async def raise_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await policy.execute(raise_type_error)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_usage(self):
        """Test RetryPolicy as a decorator."""

        @RetryPolicy(max_retries=2, initial_delay=0.01)
        async def unreliable_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return "success"

        call_count = [0]
        result = await unreliable_function()
        assert result == "success"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @RetryPolicy(max_retries=1)
        async def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestRetryPolicyEdgeCases:
    """Tests for edge cases in RetryPolicy."""

    @pytest.mark.asyncio
    async def test_max_delay_is_respected(self):
        """Test that max_delay properly caps the delay."""
        policy = RetryPolicy(
            max_retries=5,
            initial_delay=0.05,
            max_delay=0.1,
            exponential_base=10.0,  # Would grow quickly without cap
            jitter=False,
        )
        call_count = 0

        async def record_time_and_fail():
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                raise ValueError("fail")
            return "success"

        import time

        start = time.time()
        await policy.execute(record_time_and_fail)
        elapsed = time.time() - start

        assert call_count == 5
        assert elapsed < 0.5  # Initial 0.05 + 4 * 0.1 = 0.45 seconds max
