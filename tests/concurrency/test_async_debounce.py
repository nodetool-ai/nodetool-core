import asyncio
import time
from typing import Any

import pytest

from nodetool.concurrency import AsyncDebounce, async_debounce


class TestAsyncDebounce:
    """Tests for AsyncDebounce class."""

    @pytest.mark.asyncio
    async def test_debounce_waits_for_quiet_period(self) -> None:
        """Test that debounce waits until quiet period before executing."""
        calls: list[str] = []

        async def func(x: str) -> None:
            calls.append(x)

        debounced = AsyncDebounce(func, wait=0.1)

        await debounced("first")
        await debounced("second")
        await debounced("third")

        assert len(calls) == 0

        await asyncio.sleep(0.15)

        assert len(calls) == 1
        assert calls[0] == "third"

    @pytest.mark.asyncio
    async def test_debounce_replaces_previous_call(self) -> None:
        """Test that rapid calls replace each other, keeping only the last."""
        calls: list[str] = []

        async def func(x: str) -> None:
            calls.append(x)

        debounced = AsyncDebounce(func, wait=0.05)

        await debounced("first")
        await debounced("second")
        await debounced("third")

        await asyncio.sleep(0.1)

        assert len(calls) == 1
        assert calls[0] == "third"

    @pytest.mark.asyncio
    async def test_debounce_executes_with_args(self) -> None:
        """Test that debounced function receives correct arguments."""
        calls: list[tuple[tuple, dict]] = []

        async def func(*args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        debounced = AsyncDebounce(func, wait=0.05)

        await debounced("key", value=42)

        await asyncio.sleep(0.1)

        assert len(calls) == 1
        assert calls[0][0] == ("key",)
        assert calls[0][1] == {"value": 42}

    @pytest.mark.asyncio
    async def test_debounce_immediate_no_calls(self) -> None:
        """Test that without waiting, no calls are executed."""
        calls: list[str] = []

        async def func(x: str) -> None:
            calls.append(x)

        debounced = AsyncDebounce(func, wait=0.1)

        await debounced("only")
        await asyncio.sleep(0.02)

        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_debounce_multiple_quiet_periods(self) -> None:
        """Test that multiple quiet periods trigger separate executions."""
        calls: list[str] = []

        async def func(x: str) -> None:
            calls.append(x)

        debounced = AsyncDebounce(func, wait=0.05)

        await debounced("first")
        await asyncio.sleep(0.1)
        assert calls == ["first"]

        await debounced("second")
        await asyncio.sleep(0.1)
        assert calls == ["first", "second"]

    @pytest.mark.asyncio
    async def test_debounce_pending_property(self) -> None:
        """Test the pending property reflects execution state."""

        async def func(x: str) -> None:
            await asyncio.sleep(0.01)

        debounced = AsyncDebounce(func, wait=0.5)

        await debounced("test")

        assert debounced.pending is True

        await asyncio.sleep(0.6)

        assert debounced.pending is False

    @pytest.mark.asyncio
    async def test_debounce_cancel(self) -> None:
        """Test that cancel prevents execution."""
        calls: list[str] = []

        async def func(x: str) -> None:
            calls.append(x)

        debounced = AsyncDebounce(func, wait=0.2)

        await debounced("should_not_execute")

        assert debounced.pending is True

        debounced.cancel()

        assert debounced.pending is False

        await asyncio.sleep(0.3)

        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_debounce_flush(self) -> None:
        """Test that flush executes immediately."""
        calls: list[str] = []
        execution_time: float = 0.0

        async def func(x: str) -> None:
            nonlocal execution_time
            execution_time = time.monotonic()
            calls.append(x)

        debounced = AsyncDebounce(func, wait=0.5)

        await debounced("test")

        assert debounced.pending is True

        flush_future = debounced.flush()
        assert flush_future is not None

        await flush_future

        assert len(calls) == 1
        assert calls[0] == "test"

    @pytest.mark.asyncio
    async def test_debounce_flush_no_pending(self) -> None:
        """Test that flush returns None when no pending call."""

        async def func(x: str) -> None:
            pass

        debounced = AsyncDebounce(func, wait=0.1)

        await asyncio.sleep(0.2)

        assert debounced.pending is False

        flush_future = debounced.flush()

        assert flush_future is None

    @pytest.mark.asyncio
    async def test_debounce_zero_wait(self) -> None:
        """Test debounce with zero wait still defers execution."""
        calls: list[str] = []

        async def func(x: str) -> None:
            calls.append(x)

        debounced = AsyncDebounce(func, wait=0.0)

        await debounced("test")

        assert len(calls) == 0

        await asyncio.sleep(0.01)

        assert len(calls) == 1
        assert calls[0] == "test"

    @pytest.mark.asyncio
    async def test_debounce_multiple_rapid_calls_with_timing(self) -> None:
        """Test debounce with timing to verify only one call."""
        calls: list[str] = []
        start_time: float = 0.0

        async def func(x: str) -> None:
            nonlocal start_time
            if not start_time:
                start_time = time.monotonic()
            calls.append(x)

        debounced = AsyncDebounce(func, wait=0.1)

        for i in range(10):
            await debounced(f"call_{i}")

        assert len(calls) == 0

        await asyncio.sleep(0.15)

        assert len(calls) == 1
        assert calls[0] == "call_9"

    @pytest.mark.asyncio
    async def test_debounce_preserves_coroutine_behavior(self) -> None:
        """Test that debounced function can be awaited correctly."""
        results: list[int] = []

        async def func(x: int) -> int:
            await asyncio.sleep(0.01)
            results.append(x)
            return x * 2

        debounced = AsyncDebounce(func, wait=0.05)

        future = debounced(5)
        assert asyncio.iscoroutine(future)

        result = await future

        assert result is None

    @pytest.mark.asyncio
    async def test_debounce_initialization_errors(self) -> None:
        """Test that invalid initialization raises appropriate errors."""
        with pytest.raises(ValueError, match="wait must be a non-negative"):
            AsyncDebounce(lambda: None, wait=-1)

        with pytest.raises(ValueError, match="func must be callable"):
            AsyncDebounce(None, wait=0.1)  # type: ignore


class TestAsyncDebounceDecorator:
    """Tests for async_debounce decorator function."""

    @pytest.mark.asyncio
    async def test_decorator_works(self) -> None:
        """Test async_debounce as a decorator."""
        calls: list[str] = []

        @async_debounce(wait=0.05)
        async def debounced_func(x: str) -> None:
            calls.append(x)

        await debounced_func("first")
        await debounced_func("second")

        await asyncio.sleep(0.1)

        assert len(calls) == 1
        assert calls[0] == "second"

    @pytest.mark.asyncio
    async def test_decorator_with_default_wait(self) -> None:
        """Test async_debounce with default wait parameter."""
        calls: list[str] = []

        @async_debounce
        async def debounced_func(x: str) -> None:
            calls.append(x)

        await debounced_func("test")

        await asyncio.sleep(0.35)

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_decorator_functional_usage(self) -> None:
        """Test async_debounce used as a function wrapper."""
        calls: list[str] = []

        async def save(x: str) -> None:
            calls.append(x)

        debounced_save = async_debounce(save, wait=0.05)

        await debounced_save("first")
        await debounced_save("second")

        await asyncio.sleep(0.1)

        assert len(calls) == 1
        assert calls[0] == "second"

    def test_decorator_invalid_wait(self) -> None:
        """Test that invalid wait raises error."""
        with pytest.raises(ValueError, match="wait must be a non-negative"):
            async_debounce(wait=-1)


class TestAsyncDebounceEdgeCases:
    """Edge case tests for AsyncDebounce."""

    @pytest.mark.asyncio
    async def test_debounce_with_no_args(self) -> None:
        """Test debounced function with no arguments."""
        calls: list[int] = []

        async def func() -> None:
            calls.append(1)

        debounced = AsyncDebounce(func, wait=0.05)

        await debounced()

        assert len(calls) == 0

        await asyncio.sleep(0.1)

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_debounce_with_keyword_only_args(self) -> None:
        """Test debounced function with keyword-only arguments."""
        calls: list[dict] = []

        async def func(*, x: int, y: int) -> None:
            calls.append({"x": x, "y": y})

        debounced = AsyncDebounce(func, wait=0.05)

        await debounced(x=1, y=2)

        await asyncio.sleep(0.1)

        assert len(calls) == 1
        assert calls[0] == {"x": 1, "y": 2}

    @pytest.mark.asyncio
    async def test_debounce_consecutive_batches(self) -> None:
        """Test multiple batches of rapid calls."""
        calls: list[str] = []

        async def func(x: str) -> None:
            calls.append(x)

        debounced = AsyncDebounce(func, wait=0.05)

        await debounced("batch1_a")
        await debounced("batch1_b")

        await asyncio.sleep(0.1)

        await debounced("batch2_a")
        await debounced("batch2_b")
        await debounced("batch2_c")

        await asyncio.sleep(0.1)

        assert calls == ["batch1_b", "batch2_c"]
