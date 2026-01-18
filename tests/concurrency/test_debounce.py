import asyncio
import time

import pytest

from nodetool.concurrency.debounce import DebouncedCall, debounce


@pytest.fixture
async def clock():
    start = time.monotonic()
    yield lambda: time.monotonic() - start


class TestDebounce:
    async def test_basic_debounce_single_call(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            return "result"

        debounced_func = debounce(func, wait=0.05)
        result = await debounced_func()

        assert call_count == 1
        assert result == "result"

    async def test_basic_debounce_waits_before_executing(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced_func = debounce(func, wait=0.1)

        await debounced_func()

        assert call_count == 1

    async def test_basic_debounce_with_trailing(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced_func = debounce(func, wait=0.1, trailing=True)

        await debounced_func()
        await asyncio.sleep(0.05)
        await debounced_func()
        await asyncio.sleep(0.05)
        await debounced_func()

        assert call_count == 3

    async def test_debounce_leading_edge(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced_func = debounce(func, wait=0.1, leading=True)

        await debounced_func()
        assert call_count == 1

        await debounced_func()
        await asyncio.sleep(0.05)

        assert call_count == 2

        await asyncio.sleep(0.1)

        assert call_count == 2

    async def test_debounce_leading_and_trailing(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced_func = debounce(func, wait=0.1, leading=True, trailing=True)

        await debounced_func()
        assert call_count == 1

        await debounced_func()
        await asyncio.sleep(0.05)
        await debounced_func()

        assert call_count == 3

        await asyncio.sleep(0.15)

        assert call_count == 3

    async def test_debounce_max_wait(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced_func = debounce(func, wait=0.5, max_wait=0.1)

        for _ in range(5):
            await debounced_func()
            await asyncio.sleep(0.02)

        assert call_count == 5

    async def test_debounce_with_args(self):
        results = []

        async def func(x, y):
            results.append((x, y))
            return x + y

        debounced_func = debounce(func, wait=0.1)

        result = await debounced_func(1, 2)

        assert results == [(1, 2)]
        assert result == 3

    async def test_debounce_with_kwargs(self):
        results = []

        async def func(x=0, y=0):
            results.append((x, y))

        debounced_func = debounce(func, wait=0.1)

        await debounced_func(x=1, y=2)

        assert results == [(1, 2)]

    async def test_debounce_preserves_return_value(self):
        async def func():
            return "expected_result"

        debounced_func = debounce(func, wait=0.1)

        result = await debounced_func()

        assert result == "expected_result"

    async def test_debounce_invalid_wait(self):
        async def func():
            pass

        with pytest.raises(ValueError, match="wait must be a positive number"):
            debounce(func, wait=0)

        with pytest.raises(ValueError, match="wait must be a positive number"):
            debounce(func, wait=-1)

    async def test_debounce_invalid_max_wait(self):
        async def func():
            pass

        with pytest.raises(ValueError, match="max_wait must be a positive number"):
            debounce(func, wait=0.1, max_wait=0)

    async def test_debounce_both_leading_and_trailing_false(self):
        async def func():
            pass

        with pytest.raises(ValueError, match="At least one of leading or trailing must be True"):
            debounce(func, wait=0.1, leading=False, trailing=False)


class TestDebouncedCall:
    async def test_basic_usage(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced = DebouncedCall(func, wait=0.1)

        await debounced.trigger()

        assert call_count == 1

    async def test_cancel(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced = DebouncedCall(func, wait=0.1)

        await debounced.trigger()
        assert call_count == 1

    async def test_context_manager(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        async with DebouncedCall(func, wait=0.1) as debounced:
            await debounced.trigger()

        await asyncio.sleep(0.15)

        assert call_count == 1

    async def test_multiple_triggers(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced = DebouncedCall(func, wait=0.1)

        await debounced.trigger()
        await debounced.trigger()
        await debounced.trigger()

        await asyncio.sleep(0.15)

        assert call_count == 3

    async def test_with_args(self):
        results = []

        async def func(x, y):
            results.append((x, y))

        debounced = DebouncedCall(func, wait=0.1)

        await debounced.trigger(1, 2)

        await asyncio.sleep(0.15)

        assert results == [(1, 2)]

    async def test_with_kwargs(self):
        results = []

        async def func(x=0, y=0):
            results.append((x, y))

        debounced = DebouncedCall(func, wait=0.1)

        await debounced.trigger(x=1, y=2)

        await asyncio.sleep(0.15)

        assert results == [(1, 2)]

    async def test_invalid_wait(self):
        async def func():
            pass

        with pytest.raises(ValueError, match="wait must be a positive number"):
            DebouncedCall(func, wait=0)

        with pytest.raises(ValueError, match="wait must be a positive number"):
            DebouncedCall(func, wait=-1)

    async def test_both_leading_and_trailing_false(self):
        async def func():
            pass

        with pytest.raises(ValueError, match="At least one of leading or trailing must be True"):
            DebouncedCall(func, wait=0.1, leading=False, trailing=False)

    async def test_wait_time_property(self):
        async def func():
            pass

        debounced = DebouncedCall(func, wait=0.3)

        assert debounced.wait_time == 0.3

    async def test_pending_property(self):
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1

        debounced = DebouncedCall(func, wait=0.1)

        assert debounced.pending is False

        await debounced.trigger()

        assert debounced.pending is False

        await asyncio.sleep(0.05)

        assert debounced.pending is False


class TestDebounceConcurrency:
    async def test_multiple_debounced_functions_independent(self):
        call_count_1 = 0
        call_count_2 = 0

        async def func1():
            nonlocal call_count_1
            call_count_1 += 1

        async def func2():
            nonlocal call_count_2
            call_count_2 += 1

        debounced1 = debounce(func1, wait=0.1)
        debounced2 = debounce(func2, wait=0.05)

        await debounced1()
        await debounced2()
        await debounced1()
        await debounced2()

        await asyncio.sleep(0.02)
        await debounced2()

        await asyncio.sleep(0.15)

        assert call_count_1 == 2
        assert call_count_2 == 3

    async def test_debounce_during_long_running_function(self):
        call_count = 0

        async def slow_func():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.2)

        debounced_func = debounce(slow_func, wait=0.05)

        await debounced_func()

        assert call_count == 1

        await asyncio.sleep(0.1)
        await debounced_func()

        assert call_count == 2

        await asyncio.sleep(0.3)

        assert call_count == 2
