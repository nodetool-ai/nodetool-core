import asyncio

import pytest

from nodetool.concurrency.debounce import AsyncDebounce, DebouncedFunction, debounce


class TestDebouncedFunctionBasic:
    """Tests for basic debounce behavior using debounce() function."""

    @pytest.mark.asyncio
    async def test_debounce_single_call(self):
        """Test that a single call executes after the wait period."""
        call_times: list[float] = []

        async def tracked_func(value: int) -> int:
            call_times.append(asyncio.get_event_loop().time())
            return value

        debounced = debounce(0.05)(tracked_func)
        start_time = asyncio.get_event_loop().time()

        task = await debounced(42)
        result = await task

        assert result == 42
        assert len(call_times) == 1
        assert call_times[0] >= start_time + 0.05

    @pytest.mark.asyncio
    async def test_debounce_multiple_rapid_calls(self):
        """Test that rapid calls are debounced into a single call."""
        call_count = 0

        async def increment() -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return call_count

        debounced = debounce(0.1)(increment)

        tasks: list[asyncio.Task[int]] = []
        for _ in range(5):
            task = await debounced()
            tasks.append(task)
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.2)

        await asyncio.gather(*tasks, return_exceptions=True)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_debounce_last_call_wins(self):
        """Test that the last call's arguments are used."""
        results: list[int] = []

        async def store_value(value: int) -> int:
            results.append(value)
            return value

        debounced = debounce(0.05)(store_value)

        tasks: list[asyncio.Task[int]] = []
        for i in range(5):
            task = await debounced(i)
            tasks.append(task)

        await asyncio.sleep(0.2)

        await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 1
        assert results[0] == 4


class TestDebouncedFunctionFlush:
    """Tests for debounce flush behavior."""

    @pytest.mark.asyncio
    async def test_flush_executes_immediately(self):
        """Test that flush executes pending call immediately."""
        call_times: list[float] = []

        async def tracked_func(value: int) -> int:
            call_times.append(asyncio.get_event_loop().time())
            return value

        debounced = debounce(0.5)(tracked_func)
        start_time = asyncio.get_event_loop().time()

        await debounced(42)

        task = debounced.flush()
        if task:
            await task

        assert len(call_times) == 1
        assert call_times[0] < start_time + 0.1


class TestDebouncedFunctionCancel:
    """Tests for debounce cancel behavior."""

    @pytest.mark.asyncio
    async def test_cancel_prevents_execution(self):
        """Test that cancelling pending debounce prevents execution."""
        call_count = 0

        async def increment() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        debounced = debounce(0.5)(increment)

        await debounced()

        debounced.cancel()

        await asyncio.sleep(0.6)

        assert call_count == 0


class TestDebouncedFunctionPending:
    """Tests for debounce pending property."""

    @pytest.mark.asyncio
    async def test_pending_false_initially(self):
        """Test that pending is False when no call is scheduled."""

        async def dummy() -> None:
            await asyncio.sleep(0.01)

        debounced = debounce(0.5)(dummy)
        assert debounced.pending is False

    @pytest.mark.asyncio
    async def test_pending_true_before_execution(self):
        """Test that pending is True when a call is scheduled but not executed."""

        async def dummy() -> None:
            await asyncio.sleep(0.5)

        debounced = debounce(0.5)(dummy)
        await debounced()

        assert debounced.pending is True

        await asyncio.sleep(0.6)

        assert debounced.pending is False


class TestDebounceDecorator:
    """Tests for the debounce decorator function."""

    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        """Test that the decorator works correctly."""
        call_count = 0

        @debounce(0.05)
        async def increment() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        tasks: list[asyncio.Task[int]] = []
        for _ in range(3):
            task = await increment()
            tasks.append(task)

        await asyncio.sleep(0.2)

        await asyncio.gather(*tasks, return_exceptions=True)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_with_args(self):
        """Test that the decorator passes arguments correctly."""
        results: list[str] = []

        @debounce(0.05)
        async def append(value: str) -> str:
            results.append(value)
            return value

        tasks: list[asyncio.Task[str]] = []
        for c in ["a", "b", "c"]:
            task = await append(c)
            tasks.append(task)

        await asyncio.sleep(0.2)

        await asyncio.gather(*tasks, return_exceptions=True)

        assert results == ["c"]

    @pytest.mark.asyncio
    async def test_decorator_control_methods(self):
        """Test that decorator-returned function has control methods."""
        call_count = 0

        @debounce(0.5)
        async def increment() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        await increment()

        assert increment.pending is True
        increment.cancel()
        assert increment.pending is False


class TestAsyncDebounceDecorator:
    """Tests for AsyncDebounce used as a decorator."""

    @pytest.mark.asyncio
    async def test_decorator_with_wait_seconds(self):
        """Test AsyncDebounce as a decorator with wait_seconds."""
        call_count = 0

        @AsyncDebounce(wait_seconds=0.05)
        async def increment() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        tasks: list[asyncio.Task[int]] = []
        for _ in range(3):
            task = await increment()
            tasks.append(task)

        await asyncio.sleep(0.2)

        await asyncio.gather(*tasks, return_exceptions=True)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_debounce_control_methods(self):
        """Test that AsyncDebounce decorator returns function with control methods."""
        call_count = 0

        @AsyncDebounce(wait_seconds=0.5)
        async def increment() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        await increment()

        assert increment.pending is True
        increment.cancel()
        assert increment.pending is False


class TestAsyncDebounceInit:
    """Tests for AsyncDebounce initialization."""

    def test_init_valid_wait(self):
        """Test that debounce can be initialized with valid wait time."""
        debouncer = AsyncDebounce(wait_seconds=0.1)
        assert debouncer._wait_seconds == 0.1

    def test_init_invalid_wait_zero(self):
        """Test that initializing with wait_seconds=0 raises ValueError."""
        with pytest.raises(ValueError):
            AsyncDebounce(wait_seconds=0)

    def test_init_invalid_wait_negative(self):
        """Test that initializing with negative wait_seconds raises ValueError."""
        with pytest.raises(ValueError):
            AsyncDebounce(wait_seconds=-1.0)


class TestDebouncedFunctionWaitSeconds:
    """Tests for wait_seconds property."""

    @pytest.mark.asyncio
    async def test_wait_seconds_property(self):
        """Test that wait_seconds property returns correct value."""

        async def dummy() -> None:
            pass

        debounced = debounce(0.123)(dummy)
        assert debounced.wait_seconds == 0.123

    @pytest.mark.asyncio
    async def test_async_debounce_wait_seconds_property(self):
        """Test that AsyncDebounce decorator preserves wait_seconds."""

        @AsyncDebounce(wait_seconds=0.456)
        async def dummy() -> None:
            pass

        assert dummy.wait_seconds == 0.456
