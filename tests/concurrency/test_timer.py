"""Tests for AsyncTimer and related timing utilities."""
import asyncio

from nodetool.concurrency import AsyncTimer, TimerStats, async_timer, timer


class TestAsyncTimer:
    """Test AsyncTimer context manager."""

    async def test_manual_start_stop(self) -> None:
        """Test manual start and stop of timer."""
        tmr = AsyncTimer()
        assert not tmr.is_started
        assert not tmr.is_running
        assert tmr.elapsed == 0.0

        tmr.start()
        assert tmr.is_started
        assert tmr.is_running
        await asyncio.sleep(0.01)

        elapsed = tmr.stop()
        assert elapsed > 0
        assert not tmr.is_running
        assert tmr.elapsed == elapsed

    async def test_auto_start_context_manager(self) -> None:
        """Test auto-start in context manager."""
        async with AsyncTimer(auto_start=True) as tmr:
            assert tmr.is_running
            await asyncio.sleep(0.01)

        assert not tmr.is_running
        assert tmr.elapsed > 0
        assert tmr.is_started

    async def test_manual_start_in_context(self) -> None:
        """Test manual start within context manager."""
        async with AsyncTimer(auto_start=False) as tmr:
            assert not tmr.is_running
            tmr.start()
            await asyncio.sleep(0.01)

        assert not tmr.is_running
        assert tmr.elapsed > 0

    async def test_elapsed_while_running(self) -> None:
        """Test that elapsed property updates while running."""
        tmr = AsyncTimer()
        tmr.start()

        elapsed1 = tmr.elapsed
        await asyncio.sleep(0.01)
        elapsed2 = tmr.elapsed

        assert elapsed2 > elapsed1
        tmr.stop()

    async def test_reset(self) -> None:
        """Test timer reset functionality."""
        tmr = AsyncTimer()
        tmr.start()
        await asyncio.sleep(0.01)
        tmr.stop()

        assert tmr.is_started
        assert tmr.elapsed > 0

        tmr.reset()
        assert not tmr.is_started
        assert not tmr.is_running
        assert tmr.elapsed == 0.0

    async def test_double_start_raises(self) -> None:
        """Test that starting twice raises RuntimeError."""
        tmr = AsyncTimer()
        tmr.start()
        try:
            tmr.start()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "already running" in str(e)

    async def test_stop_without_start_raises(self) -> None:
        """Test that stopping without starting raises RuntimeError."""
        tmr = AsyncTimer()
        try:
            tmr.stop()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not running" in str(e)

    async def test_stats_property(self) -> None:
        """Test TimerStats from stats property."""
        async with AsyncTimer(auto_start=True) as tmr:
            await asyncio.sleep(0.01)

        stats = tmr.stats
        assert isinstance(stats, TimerStats)
        assert stats.elapsed > 0
        assert stats.start_time > 0
        assert stats.end_time is not None
        assert stats.end_time > stats.start_time

    async def test_stats_while_running(self) -> None:
        """Test TimerStats while timer is still running."""
        tmr = AsyncTimer()
        tmr.start()
        await asyncio.sleep(0.01)

        stats = tmr.stats
        assert isinstance(stats, TimerStats)
        assert stats.elapsed > 0
        assert stats.start_time > 0
        assert stats.end_time is None

    async def test_stats_not_started_raises(self) -> None:
        """Test that stats raises RuntimeError if not started."""
        tmr = AsyncTimer()
        try:
            _ = tmr.stats
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not been started" in str(e)

    async def test_repr(self) -> None:
        """Test string representation of timer."""
        tmr = AsyncTimer()
        assert "not started" in repr(tmr)

        tmr.start()
        assert "running" in repr(tmr)

        await asyncio.sleep(0.01)
        tmr.stop()
        assert "elapsed" in repr(tmr)

    async def test_timing_accuracy(self) -> None:
        """Test that timing is reasonably accurate."""
        sleep_time = 0.1
        async with AsyncTimer(auto_start=True) as tmr:
            await asyncio.sleep(sleep_time)

        # Allow 20% margin for scheduling delays
        assert tmr.elapsed >= sleep_time * 0.8
        assert tmr.elapsed <= sleep_time * 1.5


class TestTimerStats:
    """Test TimerStats class."""

    def test_timer_stats_completed(self) -> None:
        """Test TimerStats for completed timer."""
        stats = TimerStats(elapsed=1.5, start_time=100.0, end_time=101.5)
        assert stats.elapsed == 1.5
        assert stats.start_time == 100.0
        assert stats.end_time == 101.5
        assert "running" not in repr(stats)

    def test_timer_stats_running(self) -> None:
        """Test TimerStats for running timer."""
        stats = TimerStats(elapsed=1.5, start_time=100.0, end_time=None)
        assert stats.elapsed == 1.5
        assert stats.start_time == 100.0
        assert stats.end_time is None
        assert "running" in repr(stats)


class TestAsyncTimerHelper:
    """Test async_timer helper function."""

    async def test_async_timer_helper(self) -> None:
        """Test async_timer convenience function."""
        async with async_timer() as tmr:
            assert tmr.is_running
            await asyncio.sleep(0.01)

        assert tmr.elapsed > 0
        assert not tmr.is_running


class TestTimerDecorator:
    """Test timer decorator."""

    async def test_timer_decorator_basics(self) -> None:
        """Test basic timer decorator usage."""
        @timer()
        async def sample_function():
            await asyncio.sleep(0.01)
            return "result"

        result = await sample_function()
        assert result == "result"
        assert sample_function.elapsed > 0  # type: ignore

    async def test_timer_decorator_with_name(self) -> None:
        """Test timer decorator with custom name and logger."""
        logs = []

        def mock_logger(msg: str) -> None:
            logs.append(msg)

        @timer(name="CustomOperation", logger=mock_logger)
        async def sample_function():
            await asyncio.sleep(0.01)
            return "result"

        result = await sample_function()
        assert result == "result"
        assert len(logs) == 1
        assert "CustomOperation" in logs[0]
        assert "completed in" in logs[0]

    async def test_timer_decorator_default_name(self) -> None:
        """Test timer decorator uses function name by default."""
        logs = []

        def mock_logger(msg: str) -> None:
            logs.append(msg)

        @timer(logger=mock_logger)
        async def my_async_function():
            await asyncio.sleep(0.01)
            return "result"

        result = await my_async_function()
        assert result == "result"
        assert len(logs) == 1
        assert "my_async_function" in logs[0]

    async def test_timer_decorator_no_logger(self) -> None:
        """Test timer decorator without logger doesn't log."""
        @timer()
        async def sample_function():
            await asyncio.sleep(0.01)
            return "result"

        result = await sample_function()
        assert result == "result"
        assert sample_function.elapsed > 0  # type: ignore

    async def test_timer_decorator_with_args(self) -> None:
        """Test timer decorator preserves function arguments."""
        @timer()
        async def add(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a + b

        result = await add(2, 3)
        assert result == 5
        assert add.elapsed > 0  # type: ignore

    async def test_timer_decorator_with_kwargs(self) -> None:
        """Test timer decorator preserves keyword arguments."""
        @timer()
        async def greet(name: str, greeting: str = "Hello") -> str:
            await asyncio.sleep(0.01)
            return f"{greeting}, {name}!"

        result = await greet("World", greeting="Hi")
        assert result == "Hi, World!"
        assert greet.elapsed > 0  # type: ignore

    async def test_timer_decorator_multiple_calls(self) -> None:
        """Test that elapsed updates on multiple calls."""
        @timer()
        async def quick_function():
            await asyncio.sleep(0.01)
            return "done"

        await quick_function()
        first_elapsed = quick_function.elapsed  # type: ignore

        await asyncio.sleep(0.02)
        await quick_function()
        second_elapsed = quick_function.elapsed  # type: ignore

        # Both should be positive, second may vary due to scheduling
        assert first_elapsed > 0
        assert second_elapsed > 0


class TestTimerEdgeCases:
    """Test edge cases and error conditions."""

    async def test_exception_in_context(self) -> None:
        """Test that timer stops even if exception occurs."""
        tmr = AsyncTimer()

        try:
            async with tmr:
                tmr.start()
                await asyncio.sleep(0.01)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Timer should have stopped
        assert not tmr.is_running
        assert tmr.elapsed > 0

    async def test_very_short_duration(self) -> None:
        """Test timer with very short duration."""
        async with AsyncTimer(auto_start=True) as tmr:
            pass  # Immediately exit

        # Should still have some small elapsed time
        assert tmr.elapsed >= 0

    async def test_multiple_start_stop_cycles(self) -> None:
        """Test multiple start/stop cycles with reset."""
        tmr = AsyncTimer()

        for _ in range(3):
            tmr.start()
            await asyncio.sleep(0.01)
            tmr.stop()
            assert tmr.elapsed > 0
            tmr.reset()
            assert tmr.elapsed == 0

    async def test_elapsed_before_start(self) -> None:
        """Test elapsed property before timer starts."""
        tmr = AsyncTimer()
        assert tmr.elapsed == 0.0
        assert not tmr.is_started
