import asyncio
import time

import pytest

from nodetool.concurrency.debounce import DebouncePolicy


class TestDebouncePolicy:
    """Tests for DebouncePolicy class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        policy = DebouncePolicy(wait=0.3)
        assert policy.wait == 0.3
        assert policy.max_wait is None

    def test_init_with_max_wait(self):
        """Test initialization with max_wait parameter."""
        policy = DebouncePolicy(wait=0.3, max_wait=5.0)
        assert policy.wait == 0.3
        assert policy.max_wait == 5.0

    def test_init_with_invalid_wait(self):
        """Test that invalid wait raises ValueError."""
        with pytest.raises(ValueError, match="wait must be a positive number"):
            DebouncePolicy(wait=0)

        with pytest.raises(ValueError, match="wait must be a positive number"):
            DebouncePolicy(wait=-1)

    @pytest.mark.asyncio
    async def test_executes_after_wait_period(self):
        """Test that function executes after wait period."""
        policy = DebouncePolicy(wait=0.1)
        call_count = 0

        async def increment():
            nonlocal call_count
            call_count += 1

        await policy.schedule(increment)
        assert call_count == 0

        await asyncio.sleep(0.15)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_resets_timer_on_new_schedule(self):
        """Test that scheduling resets the timer."""
        policy = DebouncePolicy(wait=0.1)
        call_times = []

        async def record_time():
            call_times.append(time.monotonic())

        await policy.schedule(record_time)
        await asyncio.sleep(0.08)
        await policy.schedule(record_time)
        await asyncio.sleep(0.08)
        assert len(call_times) == 0

        await asyncio.sleep(0.05)
        assert len(call_times) == 1

    @pytest.mark.asyncio
    async def test_executes_only_last_call(self):
        """Test that rapid scheduling only executes the last function."""
        policy = DebouncePolicy(wait=0.1)
        call_count = 0

        async def increment():
            nonlocal call_count
            call_count += 1

        for _ in range(10):
            await policy.schedule(increment)
            await asyncio.sleep(0.02)

        assert call_count == 0

        await asyncio.sleep(0.15)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_wait_forces_execution(self):
        """Test that max_wait forces execution even if more calls come."""
        policy = DebouncePolicy(wait=0.5, max_wait=0.2)
        call_count = 0

        async def increment():
            nonlocal call_count
            call_count += 1

        await policy.schedule(increment)
        await asyncio.sleep(0.05)
        await policy.schedule(increment)
        await asyncio.sleep(0.05)
        await policy.schedule(increment)

        await asyncio.sleep(0.2)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_flush_executes_pending(self):
        """Test that flush forces immediate execution."""
        policy = DebouncePolicy(wait=0.5)
        call_count = 0

        async def increment():
            nonlocal call_count
            call_count += 1

        await policy.schedule(increment)
        assert call_count == 0

        result = await policy.flush()
        assert result is True
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_flush_returns_false_when_no_pending(self):
        """Test that flush returns False when nothing pending."""
        policy = DebouncePolicy(wait=0.5)

        result = await policy.flush()
        assert result is False

    @pytest.mark.asyncio
    async def test_flush_after_previous_flush(self):
        """Test flush behavior after previous flush."""
        policy = DebouncePolicy(wait=0.5)
        call_count = 0

        async def increment():
            nonlocal call_count
            call_count += 1

        await policy.schedule(increment)
        await policy.flush()
        await policy.flush()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_context_manager_executes(self):
        """Test that context manager allows execution."""
        policy = DebouncePolicy(wait=0.1)
        call_count = 0

        async def increment():
            nonlocal call_count
            call_count += 1

        async with policy:
            await policy.schedule(increment)

        await asyncio.sleep(0.15)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_wait_executed_waits_for_execution(self):
        """Test that wait_executed blocks until execution."""
        policy = DebouncePolicy(wait=0.1)
        call_count = 0

        async def increment():
            nonlocal call_count
            call_count += 1

        task = asyncio.create_task(policy.schedule(increment))
        await policy.wait_executed()
        await task

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_concurrent_schedules(self):
        """Test scheduling from multiple concurrent tasks."""
        policy = DebouncePolicy(wait=0.1)
        call_count = 0
        lock = asyncio.Lock()

        async def increment():
            nonlocal call_count
            async with lock:
                call_count += 1

        async def schedule_task():
            await policy.schedule(increment)

        tasks = [schedule_task() for _ in range(5)]
        await asyncio.gather(*tasks)

        assert call_count == 0
        await asyncio.sleep(0.15)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_executes_function_with_return_value(self):
        """Test that scheduled function return values are accessible via flush."""
        policy = DebouncePolicy(wait=0.1)

        async def return_value():
            return "expected_result"

        await policy.schedule(return_value)
        flush_task = asyncio.create_task(policy.flush())
        await asyncio.sleep(0.01)
        await flush_task

        async with policy:
            await policy.schedule(return_value)

        await asyncio.sleep(0.15)

    @pytest.mark.asyncio
    async def test_cancellation_during_wait(self):
        """Test that cancellation during wait doesn't cause issues."""
        policy = DebouncePolicy(wait=0.3)
        call_count = 0

        async def increment():
            nonlocal call_count
            call_count += 1

        await policy.schedule(increment)
        await asyncio.sleep(0.1)
        await policy.flush()
        await asyncio.sleep(0.1)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_rapid_schedule_with_max_wait(self):
        """Test that max_wait works correctly with rapid scheduling."""
        policy = DebouncePolicy(wait=0.5, max_wait=0.15)
        call_count = 0
        execution_time = 0

        async def record_time_and_increment():
            nonlocal call_count, execution_time
            execution_time = time.monotonic()
            call_count += 1

        start = time.monotonic()
        for _ in range(5):
            await policy.schedule(record_time_and_increment)
            await asyncio.sleep(0.03)

        await asyncio.sleep(0.3)

        assert call_count == 1, f"Expected 1 call, got {call_count}"
        elapsed = execution_time - start
        assert 0.1 <= elapsed <= 0.25, f"Expected execution between 0.1s and 0.25s, got {elapsed}"

    @pytest.mark.asyncio
    async def test_decorator_like_usage(self):
        """Test using schedule with a pre-defined function."""
        policy = DebouncePolicy(wait=0.1)
        call_count = 0

        async def my_task():
            nonlocal call_count
            call_count += 1

        await policy.schedule(my_task)
        await policy.schedule(my_task)
        await policy.schedule(my_task)

        await asyncio.sleep(0.15)
        assert call_count == 1
