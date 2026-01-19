import asyncio
from collections.abc import Awaitable, Callable

import pytest

from nodetool.concurrency.debounce import AsyncDebounce, debounce


class TestAsyncDebounce:
    """Test cases for AsyncDebounce utility."""

    @pytest.mark.asyncio
    async def test_debounce_returns_result(self) -> None:
        """Test that debounced function returns the correct result."""

        async def func() -> str:
            await asyncio.sleep(0.01)
            return "expected_result"

        debounced = AsyncDebounce(func, wait=0.05)

        result = await debounced()

        assert result == "expected_result"

    @pytest.mark.asyncio
    async def test_debounce_error_in_function(self) -> None:
        """Test that errors in wrapped function are propagated."""

        async def func() -> None:
            raise ValueError("test error")

        debounced = AsyncDebounce(func, wait=0.05)

        with pytest.raises(ValueError, match="test error"):
            await debounced()

    @pytest.mark.asyncio
    async def test_debounce_with_args(self) -> None:
        """Test that arguments are passed correctly to debounced function."""
        received: list[tuple[tuple[int, ...], dict[str, int]]] = []

        async def func(*args: int, **kwargs: int) -> None:
            received.append((args, kwargs))

        debounced = AsyncDebounce(func, wait=0.05)

        await debounced(1, 2, 3, a=4, b=5)
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert received[0] == ((1, 2, 3), {"a": 4, "b": 5})

    @pytest.mark.asyncio
    async def test_debounce_waits_before_execution(self) -> None:
        """Test that debounce waits before executing."""
        start_time = asyncio.get_event_loop().time()
        execution_time = 0.0

        async def func() -> None:
            nonlocal execution_time
            await asyncio.sleep(0.01)
            execution_time = asyncio.get_event_loop().time()

        debounced = AsyncDebounce(func, wait=0.1)

        debounced_task = asyncio.create_task(debounced())

        await asyncio.sleep(0.05)

        assert execution_time == 0.0, "Function should not execute during wait period"

        await debounced_task

        assert execution_time > start_time + 0.1, "Function should execute after wait period"


class TestDebounceDecorator:
    """Test cases for the debounce decorator function."""

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self) -> None:
        """Test that decorator preserves the wrapped function's name."""

        @debounce(wait=0.1)
        async def my_function() -> None:
            pass

        assert my_function._func.__name__ == "my_function"

    @pytest.mark.asyncio
    async def test_decorator_with_args(self) -> None:
        """Test decorator with function that has arguments."""

        @debounce(wait=0.05)
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(1, 2)

        assert result == 3

    @pytest.mark.asyncio
    async def test_decorator_single_call(self) -> None:
        """Test that single call executes."""
        call_count = 0

        @debounce(wait=0.05)
        async def func() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        await func()
        await asyncio.sleep(0.15)

        assert call_count == 1
