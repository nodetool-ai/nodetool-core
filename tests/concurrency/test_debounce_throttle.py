import asyncio

import pytest

from nodetool.concurrency.debounce_throttle import (
    AdaptiveThrottle,
    AsyncDebounce,
    AsyncThrottle,
)


class TestAsyncDebounce:
    """Tests for AsyncDebounce class."""

    @pytest.mark.asyncio
    async def test_single_execution(self):
        """Test that a single call executes after delay."""
        debounce = AsyncDebounce(delay=0.1)
        call_count = 0

        async def test_func():
            nonlocal call_count
            call_count += 1
            return "result"

        # Create task for non-blocking execution
        task = asyncio.create_task(debounce.execute(test_func))
        await task

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_calls_only_last_executes(self):
        """Test that rapid calls only execute the last one."""
        debounce = AsyncDebounce(delay=0.15)
        call_count = 0
        results = []

        async def test_func(value):
            nonlocal call_count
            call_count += 1
            results.append(value)
            return value

        # Rapid calls - each creates a task that will be cancelled/replaced
        _ = asyncio.create_task(debounce.execute(lambda: test_func("first")))  # noqa: RUF006
        await asyncio.sleep(0.02)  # Small delay

        _ = asyncio.create_task(debounce.execute(lambda: test_func("second")))  # noqa: RUF006
        await asyncio.sleep(0.02)  # Small delay

        task3 = asyncio.create_task(debounce.execute(lambda: test_func("third")))

        # Wait for debounce to complete
        await task3

        # Wait a bit more to ensure no more executions
        await asyncio.sleep(0.2)

        # Only the last call should execute
        assert call_count == 1
        assert results == ["third"]

    @pytest.mark.asyncio
    async def test_cancel_pending_execution(self):
        """Test canceling a pending debounced execution."""
        debounce = AsyncDebounce(delay=0.5)
        call_count = 0

        async def test_func():
            nonlocal call_count
            call_count += 1
            return "result"

        # Start execution
        _ = asyncio.create_task(debounce.execute(test_func))  # noqa: RUF006

        # Give the event loop a chance to schedule the timer
        await asyncio.sleep(0.01)

        # Cancel immediately
        cancelled = await debounce.cancel()
        assert cancelled is True

        # Wait to ensure cancellation worked
        await asyncio.sleep(0.1)

        assert call_count == 0

    @pytest.mark.asyncio
    async def test_cancel_returns_false_when_no_pending(self):
        """Test cancel returns False when no task is pending."""
        debounce = AsyncDebounce(delay=0.1)

        cancelled = await debounce.cancel()
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_is_pending(self):
        """Test is_pending method."""
        debounce = AsyncDebounce(delay=0.2)

        async def test_func():
            return "result"

        # Start execution
        task = asyncio.create_task(debounce.execute(test_func))

        # Give the event loop a chance to schedule
        await asyncio.sleep(0.01)

        # Should be pending
        assert debounce.is_pending() is True

        # Wait for completion
        await task

        # Should not be pending
        assert debounce.is_pending() is False

    @pytest.mark.asyncio
    async def test_invalid_delay(self):
        """Test that invalid delay raises ValueError."""
        with pytest.raises(ValueError, match="delay must be positive"):
            AsyncDebounce(delay=0)

        with pytest.raises(ValueError, match="delay must be positive"):
            AsyncDebounce(delay=-0.1)

    @pytest.mark.asyncio
    async def test_concurrent_debounces(self):
        """Test multiple concurrent debounced operations."""
        debounce1 = AsyncDebounce(delay=0.1)
        debounce2 = AsyncDebounce(delay=0.1)

        count1 = 0
        count2 = 0

        async def func1():
            nonlocal count1
            count1 += 1

        async def func2():
            nonlocal count2
            count2 += 1

        await asyncio.gather(
            debounce1.execute(func1),
            debounce2.execute(func2),
        )

        assert count1 == 1
        assert count2 == 1


class TestAsyncThrottle:
    """Tests for AsyncThrottle class."""

    @pytest.mark.asyncio
    async def test_first_call_executes_immediately(self):
        """Test that the first call executes immediately."""
        throttle = AsyncThrottle(interval=1.0)
        call_count = 0

        async def test_func():
            nonlocal call_count
            call_count += 1
            return "result"

        result = await throttle.execute(test_func)
        assert result == "result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_throttled_call_skipped(self):
        """Test that calls within interval are skipped."""
        throttle = AsyncThrottle(interval=0.2)
        call_count = 0

        async def test_func():
            nonlocal call_count
            call_count += 1
            return "result"

        # First call executes
        result1 = await throttle.execute(test_func)
        assert result1 == "result"
        assert call_count == 1

        # Immediate second call is throttled
        result2 = await throttle.execute(test_func)
        assert result2 is None
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_execution_after_interval(self):
        """Test that execution resumes after interval passes."""
        throttle = AsyncThrottle(interval=0.1)
        call_count = 0

        async def test_func():
            nonlocal call_count
            call_count += 1
            return "result"

        # First call
        await throttle.execute(test_func)
        assert call_count == 1

        # Wait for interval to pass
        await asyncio.sleep(0.15)

        # Second call should execute
        await throttle.execute(test_func)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_wait_mode(self):
        """Test throttle with wait mode enabled."""
        throttle = AsyncThrottle(interval=0.1)
        call_count = 0

        async def test_func():
            nonlocal call_count
            call_count += 1
            return "result"

        # First call executes immediately
        result1 = await throttle.execute(test_func)
        assert result1 == "result"
        assert call_count == 1

        # Second call waits
        result2 = await throttle.execute(test_func, skip_if_throttled=False)
        assert result2 == "result"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting the throttle timer."""
        throttle = AsyncThrottle(interval=0.5)
        call_count = 0

        async def test_func():
            nonlocal call_count
            call_count += 1
            return "result"

        # First call
        await throttle.execute(test_func)
        assert call_count == 1

        # Reset throttle
        await throttle.reset()

        # Should execute immediately after reset
        await throttle.execute(test_func)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_can_execute(self):
        """Test can_execute method."""
        throttle = AsyncThrottle(interval=0.1)

        # Initially should be able to execute
        assert throttle.can_execute() is True

        async def test_func():
            return "result"

        # After execution
        await throttle.execute(test_func)

        # Should be throttled
        assert throttle.can_execute() is False

        # Wait for interval
        await asyncio.sleep(0.15)

        # Should be able to execute again
        assert throttle.can_execute() is True

    @pytest.mark.asyncio
    async def test_invalid_interval(self):
        """Test that invalid interval raises ValueError."""
        with pytest.raises(ValueError, match="interval must be positive"):
            AsyncThrottle(interval=0)

        with pytest.raises(ValueError, match="interval must be positive"):
            AsyncThrottle(interval=-0.1)

    @pytest.mark.asyncio
    async def test_concurrent_throttles(self):
        """Test multiple concurrent throttled operations."""
        throttle1 = AsyncThrottle(interval=0.1)
        throttle2 = AsyncThrottle(interval=0.1)

        count1 = 0
        count2 = 0

        async def func1():
            nonlocal count1
            count1 += 1
            return "result1"

        async def func2():
            nonlocal count2
            count2 += 1
            return "result2"

        results = await asyncio.gather(
            throttle1.execute(func1),
            throttle2.execute(func2),
        )

        assert results[0] == "result1"
        assert results[1] == "result2"
        assert count1 == 1
        assert count2 == 1


class TestAdaptiveThrottle:
    """Tests for AdaptiveThrottle class."""

    @pytest.mark.asyncio
    async def test_success_reduces_interval(self):
        """Test that successful calls reduce the interval."""
        throttle = AdaptiveThrottle(
            min_interval=0.05,
            max_interval=1.0,
            initial_interval=0.2,
            recovery_multiplier=0.5,
        )

        async def success_func():
            return "success"

        # First execution
        result = await throttle.execute(success_func)
        assert result == "success"

        # Interval should have decreased
        current_interval = throttle.get_current_interval()
        assert current_interval < 0.2
        assert current_interval >= 0.05

    @pytest.mark.asyncio
    async def test_failure_increases_interval(self):
        """Test that failed calls increase the interval."""
        throttle = AdaptiveThrottle(
            min_interval=0.05,
            max_interval=2.0,
            initial_interval=0.2,
            backoff_multiplier=2.0,
        )

        async def failing_func():
            raise ValueError("test error")

        # First failure
        with pytest.raises(ValueError, match="test error"):
            await throttle.execute(failing_func)

        # Interval should have increased
        current_interval = throttle.get_current_interval()
        assert current_interval > 0.2
        assert current_interval <= 2.0

    @pytest.mark.asyncio
    async def test_interval_bounds(self):
        """Test that interval respects min and max bounds."""
        throttle = AdaptiveThrottle(
            min_interval=0.1,
            max_interval=1.0,
            initial_interval=0.5,
            recovery_multiplier=0.1,  # Very aggressive reduction
            backoff_multiplier=10.0,  # Very aggressive increase
        )

        async def success_func():
            return "success"

        async def failing_func():
            raise ValueError("error")

        # Test minimum bound
        for _ in range(5):
            await throttle.execute(success_func)
        assert throttle.get_current_interval() >= 0.1

        # Reset
        await throttle.reset()

        # Test maximum bound
        for _ in range(5):
            try:
                await throttle.execute(failing_func)
            except ValueError:
                pass
        assert throttle.get_current_interval() <= 1.0

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting adaptive throttle."""
        throttle = AdaptiveThrottle(
            min_interval=0.1,
            max_interval=2.0,
            initial_interval=0.5,
            backoff_multiplier=2.0,
            recovery_multiplier=0.8,
        )

        async def failing_func():
            raise ValueError("error")

        # Cause interval to increase
        try:
            await throttle.execute(failing_func)
        except ValueError:
            pass

        assert throttle.get_current_interval() > 0.5

        # Reset
        await throttle.reset()

        # Should be back to initial
        assert throttle.get_current_interval() == 0.5

    @pytest.mark.asyncio
    async def test_invalid_parameters(self):
        """Test that invalid parameters raise ValueError."""
        # Negative intervals
        with pytest.raises(ValueError, match="intervals must be positive"):
            AdaptiveThrottle(min_interval=0)

        # min > max
        with pytest.raises(ValueError, match="min_interval must be <= max_interval"):
            AdaptiveThrottle(min_interval=1.0, max_interval=0.5)

        # initial outside bounds
        with pytest.raises(ValueError, match="initial_interval must be between"):
            AdaptiveThrottle(
                min_interval=0.1, max_interval=1.0, initial_interval=2.0
            )

    @pytest.mark.asyncio
    async def test_get_current_interval(self):
        """Test get_current_interval method."""
        throttle = AdaptiveThrottle(
            min_interval=0.1,
            max_interval=1.0,
            initial_interval=0.5,
        )

        assert throttle.get_current_interval() == 0.5

        async def success_func():
            return "success"

        await throttle.execute(success_func)

        # Interval should have changed
        new_interval = throttle.get_current_interval()
        assert new_interval != 0.5
        assert 0.1 <= new_interval <= 1.0


class TestIntegrationScenarios:
    """Integration tests for debounce and throttle."""

    @pytest.mark.asyncio
    async def test_debounce_and_throttle(self):
        """Test using debounce and throttle together."""
        debounce = AsyncDebounce(delay=0.1)
        throttle = AsyncThrottle(interval=0.15)

        call_count = 0

        async def test_func():
            nonlocal call_count
            call_count += 1
            return f"call-{call_count}"

        # Debounce rapid calls, then throttle the result
        _ = asyncio.create_task(  # noqa: RUF006
            debounce.execute(lambda: throttle.execute(test_func, skip_if_throttled=False))
        )

        await asyncio.sleep(0.3)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_adaptive_throttle_with_retry(self):
        """Test adaptive throttle with a simulated unreliable service."""
        throttle = AdaptiveThrottle(
            min_interval=0.05,
            max_interval=1.0,
            initial_interval=0.1,
            backoff_multiplier=2.0,
            recovery_multiplier=0.5,
        )

        attempt = 0

        async def unreliable_func():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ValueError("service unavailable")
            return "success"

        # First two attempts fail
        for _ in range(2):
            try:
                await throttle.execute(unreliable_func)
            except ValueError:
                pass

        # Interval should have increased
        assert throttle.get_current_interval() > 0.1

        # Third attempt succeeds
        result = await throttle.execute(unreliable_func)
        assert result == "success"

        # Interval should have decreased
        final_interval = throttle.get_current_interval()
        assert final_interval < throttle._max_interval
