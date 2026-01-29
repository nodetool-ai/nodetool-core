import asyncio
import time

import pytest

from nodetool.concurrency import DebouncedFunc, debounce


class TestDebouncedFunc:
    """Tests for DebouncedFunc class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""

        async def func():
            pass

        debounced = DebouncedFunc(func, wait=0.5)
        assert debounced.wait == 0.5

    def test_init_with_invalid_wait(self):
        """Test that invalid wait raises ValueError."""

        async def func():
            pass

        with pytest.raises(ValueError, match="wait must be a positive number"):
            DebouncedFunc(func, wait=0)

        with pytest.raises(ValueError, match="wait must be a positive number"):
            DebouncedFunc(func, wait=-1)

    def test_init_with_default_wait(self):
        """Test initialization with default wait time."""

        async def func():
            pass

        debounced = DebouncedFunc(func)
        assert debounced.wait == 0.3

    def test_pending_property_no_call(self):
        """Test pending property when no call has been made."""

        async def func():
            pass

        debounced = DebouncedFunc(func)
        assert debounced.pending is False

    @pytest.mark.asyncio
    async def test_single_call_executes_after_wait(self):
        """Test that a single call executes after wait period."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced = DebouncedFunc(func, wait=0.1)

        await debounced()
        assert call_count == 0

        await asyncio.sleep(0.15)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_calls_only_last_executes(self):
        """Test that rapid calls only execute the last one."""
        call_count = 0
        last_value = None

        async def func(value):
            nonlocal call_count, last_value
            call_count += 1
            last_value = value

        debounced = DebouncedFunc(func, wait=0.2)

        await debounced("value1")
        await debounced("value2")
        await debounced("value3")

        await asyncio.sleep(0.3)

        assert call_count == 1
        assert last_value == "value3"

    @pytest.mark.asyncio
    async def test_arguments_passed_correctly(self):
        """Test that arguments are passed to the function correctly."""
        received_args = None
        received_kwargs = None

        async def func(*args, **kwargs):
            nonlocal received_args, received_kwargs
            received_args = args
            received_kwargs = kwargs

        debounced = DebouncedFunc(func, wait=0.1)
        await debounced("arg1", "arg2", key1="value1", key2="value2")

        await asyncio.sleep(0.2)

        assert received_args == ("arg1", "arg2")
        assert received_kwargs == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_flush_executes_pending(self):
        """Test that flush executes any pending call immediately."""
        call_count = 0
        start_time = None

        async def func():
            nonlocal call_count, start_time
            if start_time is None:
                start_time = time.time()
            call_count += 1

        debounced = DebouncedFunc(func, wait=1.0)

        await debounced()
        assert call_count == 0

        await debounced.flush()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_flush_with_no_pending(self):
        """Test flush when no call is pending."""

        async def func():
            pass

        debounced = DebouncedFunc(func, wait=0.1)

        result = await debounced.flush()
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_discards_pending(self):
        """Test that cancel discards pending call."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced = DebouncedFunc(func, wait=1.0)

        await debounced()
        assert call_count == 0

        debounced.cancel()
        await asyncio.sleep(0.2)

        assert call_count == 0

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test that exceptions are handled correctly."""
        error_raised = False

        async def func():
            nonlocal error_raised
            error_raised = True
            raise ValueError("test error")

        debounced = DebouncedFunc(func, wait=0.1)

        await debounced()
        await asyncio.sleep(0.2)

        assert error_raised is True

    @pytest.mark.asyncio
    async def test_pending_property_during_wait(self):
        """Test pending property during wait period."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced = DebouncedFunc(func, wait=0.1)
        await debounced()

        assert debounced.pending is True

        await asyncio.sleep(0.2)
        assert debounced.pending is False
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_zero_wait_time(self):
        """Test debounce with very short wait time still debounces."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced = DebouncedFunc(func, wait=0.001)
        await debounced()
        await debounced()

        await asyncio.sleep(0.05)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_no_arguments(self):
        """Test debounced function with no arguments."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced = DebouncedFunc(func, wait=0.1)

        await debounced()
        await debounced()

        await asyncio.sleep(0.2)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_keyword_only_arguments(self):
        """Test debounced function with keyword-only arguments."""
        received_kwargs = None

        async def func(*, key1, key2):
            nonlocal received_kwargs
            received_kwargs = {"key1": key1, "key2": key2}

        debounced = DebouncedFunc(func, wait=0.1)
        await debounced(key1="value1", key2="value2")

        await asyncio.sleep(0.2)

        assert received_kwargs == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_exception_doesnt_break_debounce(self):
        """Test that exception doesn't break debounce functionality."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("expected error")

        debounced = DebouncedFunc(func, wait=0.1)
        await debounced()

        with pytest.raises(ValueError):
            await debounced.flush()

        call_count = 0

        async def func2():
            nonlocal call_count
            call_count += 1

        debounced2 = DebouncedFunc(func2, wait=0.1)
        await debounced2()
        await asyncio.sleep(0.2)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_concurrent_calls(self):
        """Test behavior with multiple concurrent debounced calls."""
        call_count = 0
        results = []

        async def func(value):
            nonlocal call_count
            call_count += 1
            results.append(value)

        debounced = DebouncedFunc(func, wait=0.2)

        tasks = [asyncio.create_task(debounced(i)) for i in range(5)]

        await asyncio.sleep(0.3)

        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert call_count == 1
        assert len(results) == 1
        assert results[0] == 4

    @pytest.mark.asyncio
    async def test_rapid_successive_calls(self):
        """Test many rapid calls in succession."""
        call_count = 0
        last_value = None

        async def func(value):
            nonlocal call_count, last_value
            call_count += 1
            last_value = value

        debounced = DebouncedFunc(func, wait=0.1)

        for i in range(100):
            await debounced(i)

        await asyncio.sleep(0.2)

        assert call_count == 1
        assert last_value == 99

    @pytest.mark.asyncio
    async def test_wait_between_groups(self):
        """Test that separate groups of calls execute separately."""
        call_count = 0
        values = []

        async def func(value):
            nonlocal call_count, values
            call_count += 1
            values.append(value)

        debounced = DebouncedFunc(func, wait=0.1)

        await debounced(1)
        await debounced(2)
        await asyncio.sleep(0.15)

        await debounced(3)
        await debounced(4)
        await asyncio.sleep(0.15)

        assert call_count == 2
        assert values == [2, 4]
