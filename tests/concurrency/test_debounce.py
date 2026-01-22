import asyncio
import time

import pytest

from nodetool.concurrency.debounce import AsyncDebounce, DebounceError, DebounceGroup


class TestAsyncDebounce:
    """Tests for AsyncDebounce class."""

    def test_basic_debounce(self) -> None:
        """Test that debounce delays execution."""
        results: list[str] = []

        async def add_value(value: str) -> None:
            results.append(value)

        debounced = AsyncDebounce(wait_seconds=0.05)(add_value)

        async def run_test() -> None:
            await debounced("a")
            await debounced("b")
            await debounced("c")
            assert len(results) == 0
            await asyncio.sleep(0.1)
            assert results == ["c"]

        asyncio.run(run_test())

    def test_debounce_leading_edge(self) -> None:
        """Test that leading edge executes immediately on first call."""
        results: list[str] = []

        async def add_value(value: str) -> None:
            results.append(value)

        debounced = AsyncDebounce(wait_seconds=0.05, leading=True)(add_value)

        async def run_test() -> None:
            # Leading edge should execute immediately
            await debounced("a")
            assert results == ["a"]

            # Subsequent calls are debounced and don't execute immediately
            await debounced("b")
            await debounced("c")
            assert results == ["a"]

            # After quiet period, the last call executes (trailing edge)
            await asyncio.sleep(0.1)
            assert "c" in results

        asyncio.run(run_test())

    def test_debounce_flush(self) -> None:
        """Test that flush executes pending call immediately."""
        results: list[str] = []

        async def add_value(value: str) -> None:
            results.append(value)

        debounced = AsyncDebounce(wait_seconds=0.5)(add_value)

        async def run_test() -> None:
            await debounced("a")
            await debounced("b")
            assert len(results) == 0
            result = await debounced.flush()
            assert results == ["b"]
            assert result == [None]

        asyncio.run(run_test())

    def test_debounce_cancel(self) -> None:
        """Test that cancel clears pending execution."""
        results: list[str] = []

        async def add_value(value: str) -> None:
            results.append(value)

        debounced = AsyncDebounce(wait_seconds=0.5)(add_value)

        async def run_test() -> None:
            await debounced("a")
            debounced.cancel()
            await asyncio.sleep(0.1)
            assert len(results) == 0

        asyncio.run(run_test())

    def test_debounce_reset(self) -> None:
        """Test that reset clears state."""
        results: list[str] = []

        async def add_value(value: str) -> None:
            results.append(value)

        debouncer = AsyncDebounce(wait_seconds=0.5)
        debounced = debouncer(add_value)

        async def run_test() -> None:
            await debounced("a")
            debounced.reset()
            assert debouncer.call_count == 0
            assert not debouncer.is_pending

        asyncio.run(run_test())

    def test_debounce_call_count(self) -> None:
        """Test that call_count tracks debounced calls."""
        call_count = 0

        async def increment() -> None:
            nonlocal call_count
            call_count += 1

        debouncer = AsyncDebounce(wait_seconds=0.05)
        debounced = debouncer(increment)

        async def run_test() -> None:
            for _ in range(5):
                await debounced()
            assert debouncer.call_count == 5
            await asyncio.sleep(0.1)
            assert call_count == 1

        asyncio.run(run_test())

    def test_debounce_max_wait(self) -> None:
        """Test that max_wait forces execution."""
        results: list[str] = []

        async def add_value(value: str) -> None:
            results.append(value)

        debounced = AsyncDebounce(wait_seconds=0.5, max_wait=0.03)(add_value)

        async def run_test() -> None:
            # First call sets execute_at
            await debounced("a")
            # Second call within max_wait should not trigger immediate execution
            await asyncio.sleep(0.01)
            await debounced("b")
            await asyncio.sleep(0.01)
            await debounced("c")
            # Wait for max_wait to trigger
            await asyncio.sleep(0.05)
            # Should have executed once due to max_wait
            assert len(results) == 1

        asyncio.run(run_test())

    def test_debounce_properties(self) -> None:
        """Test that properties return correct values."""
        debouncer = AsyncDebounce(wait_seconds=0.5)
        debouncer(lambda: None)
        assert debouncer.wait_seconds == 0.5
        assert debouncer.call_count == 0
        assert not debouncer.is_pending

    def test_debounce_invalid_wait_seconds(self) -> None:
        """Test that invalid wait_seconds raises ValueError."""
        with pytest.raises(ValueError, match="wait_seconds must be a positive number"):
            AsyncDebounce(wait_seconds=0)(lambda: None)

        with pytest.raises(ValueError, match="wait_seconds must be a positive number"):
            AsyncDebounce(wait_seconds=-1)(lambda: None)

    def test_debounce_invalid_edges(self) -> None:
        """Test that invalid edge combination raises ValueError."""
        with pytest.raises(ValueError, match="At least one of leading or trailing must be True"):
            AsyncDebounce(wait_seconds=0.5, leading=False, trailing=False)(lambda: None)

    def test_debounce_function_return_value(self) -> None:
        """Test that function return values are preserved."""

        async def return_value(x: int) -> int:
            return x * 2

        debounced = AsyncDebounce(wait_seconds=0.05)(return_value)

        async def run_test() -> None:
            result = await debounced(21)
            await asyncio.sleep(0.1)
            # Trailing edge returns None, only leading edge would return the value
            # This is expected behavior for trailing edge debouncing
            assert result is None

        asyncio.run(run_test())


class TestDebounceGroup:
    """Tests for DebounceGroup class."""

    def test_group_basic(self) -> None:
        """Test that group debounces multiple functions together."""
        results: list[str] = []

        async def add_value(value: str) -> None:
            results.append(value)

        group = DebounceGroup(wait_seconds=0.05)
        func1 = group(add_value)
        func2 = group(add_value)

        async def run_test() -> None:
            await func1("a")
            await func1("b")
            await func2("c")
            assert len(results) == 0
            await asyncio.sleep(0.1)
            # Both functions should execute once
            assert len(results) == 2
            assert "b" in results
            assert "c" in results

        asyncio.run(run_test())

    def test_group_flush(self) -> None:
        """Test that group flush executes all pending calls."""
        results: list[str] = []

        async def add_value(value: str) -> None:
            results.append(value)

        group = DebounceGroup(wait_seconds=0.5)
        func1 = group(add_value)
        func2 = group(add_value)

        async def run_test() -> None:
            await func1("a")
            await func2("b")
            await group.flush()
            assert len(results) == 2
            assert "a" in results
            assert "b" in results

        asyncio.run(run_test())

    def test_group_cancel(self) -> None:
        """Test that group cancel clears pending calls."""
        results: list[str] = []

        async def add_value(value: str) -> None:
            results.append(value)

        group = DebounceGroup(wait_seconds=0.5)
        func = group(add_value)

        async def run_test() -> None:
            await func("a")
            group.cancel()
            await asyncio.sleep(0.1)
            assert len(results) == 0

        asyncio.run(run_test())


class TestDebounceError:
    """Tests for DebounceError exception."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = DebounceError()
        assert "Function call rejected during debounce cooldown" in str(error)

    def test_custom_message(self) -> None:
        """Test custom error message."""
        error = DebounceError("Custom message")
        assert str(error) == "Custom message"


@pytest.mark.asyncio
async def test_debounce_concurrent_calls() -> None:
    """Test debounce with concurrent calls from multiple sources."""
    results: list[int] = []

    async def process(n: int) -> None:
        results.append(n)

    debounced = AsyncDebounce(wait_seconds=0.02)(process)

    async with asyncio.TaskGroup() as tg:
        for i in range(10):
            tg.create_task(debounced(i))

    await asyncio.sleep(0.05)
    assert len(results) == 1
    assert results[0] == 9


@pytest.mark.asyncio
async def test_debounce_timing() -> None:
    """Test that debounce timing is approximately correct."""
    timestamps: list[float] = []

    async def record() -> None:
        timestamps.append(time.monotonic())

    debounced = AsyncDebounce(wait_seconds=0.02)(record)

    await debounced()
    await debounced()
    start = time.monotonic()
    await asyncio.sleep(0.03)
    end = time.monotonic()

    assert len(timestamps) == 1
    assert start <= timestamps[0] <= end


@pytest.mark.asyncio
async def test_debounce_no_function() -> None:
    """Test debounce behavior when no function is called."""
    debounced = AsyncDebounce(wait_seconds=0.05)(lambda: None)

    result = await debounced.flush()
    assert result == []


@pytest.mark.asyncio
async def test_debounce_attributes() -> None:
    """Test that debounced functions have attributes."""

    async def my_func() -> None:
        pass

    debounced = AsyncDebounce(wait_seconds=0.05)(my_func)

    assert hasattr(debounced, "_is_debounced")
    assert debounced._is_debounced is True
    assert hasattr(debounced, "flush")
    assert hasattr(debounced, "cancel")
    assert hasattr(debounced, "reset")


@pytest.mark.asyncio
async def test_debounce_group_attributes() -> None:
    """Test that grouped functions have attributes."""

    async def my_func() -> None:
        pass

    group = DebounceGroup(wait_seconds=0.05)
    wrapped = group(my_func)

    assert hasattr(wrapped, "_is_grouped")
    assert wrapped._is_grouped is True
    assert hasattr(wrapped, "flush")
    assert hasattr(wrapped, "cancel")


@pytest.mark.asyncio
async def test_debounce_pending_count() -> None:
    """Test that pending_count returns correct value."""
    debouncer = AsyncDebounce(wait_seconds=0.5)
    debounced = debouncer(lambda: None)

    await debounced("a")
    await debounced("b")
    assert debouncer.pending_count == 1


@pytest.mark.asyncio
async def test_debounce_different_functions() -> None:
    """Test debounce with different functions."""
    results_a: list[str] = []
    results_b: list[str] = []

    async def func_a(value: str) -> None:
        results_a.append(value)

    async def func_b(value: str) -> None:
        results_b.append(value)

    debounced_a = AsyncDebounce(wait_seconds=0.05)(func_a)
    debounced_b = AsyncDebounce(wait_seconds=0.05)(func_b)

    await debounced_a("a1")
    await debounced_a("a2")
    await debounced_b("b1")
    await debounced_b("b2")

    await asyncio.sleep(0.1)

    assert results_a == ["a2"]
    assert results_b == ["b2"]
