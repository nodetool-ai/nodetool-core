"""
Async timing utilities for measuring execution duration of async operations.

This module provides a context manager and decorator for measuring how long
async operations take, useful for performance monitoring and debugging.
"""
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class TimerStats:
    """
    Statistics collected by AsyncTimer.

    Attributes:
        elapsed: Elapsed time in seconds.
        start_time: Monotonic clock time when timing started.
        end_time: Monotonic clock time when timing ended (None if still running).
    """

    def __init__(self, elapsed: float, start_time: float, end_time: float | None):
        self.elapsed = elapsed
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self) -> str:
        if self.end_time is None:
            return f"TimerStats(elapsed={self.elapsed:.6f}s, running)"
        return f"TimerStats(elapsed={self.elapsed:.6f}s)"


class AsyncTimer:
    """
    Context manager for measuring execution time of async code blocks.

    Provides high-resolution timing using monotonic clock, suitable for
    measuring execution time of async operations.

    Example:
        timer = AsyncTimer()
        async with timer:
            await some_async_operation()

        print(f"Operation took {timer.elapsed:.3f} seconds")

    Example with auto-start:
        async with AsyncTimer() as timer:
            await some_async_operation()

        print(f"Operation took {timer.elapsed:.3f} seconds")

    Example accessing stats:
        async with AsyncTimer() as timer:
            await some_async_operation()

        stats = timer.stats
        print(f"Start: {stats.start_time}, End: {stats.end_time}")
        print(f"Elapsed: {stats.elapsed:.6f} seconds")
    """

    def __init__(self, auto_start: bool = False):
        """
        Initialize the timer.

        Args:
            auto_start: If True, starts timing immediately on entry.
                        If False, you must call start() explicitly.
        """
        self._auto_start = auto_start
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._elapsed: float | None = None
        self._running = False

    @property
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.

        Returns:
            Elapsed time in seconds. If timer is still running, returns
            time elapsed so far. If timer hasn't started, returns 0.0.

        Raises:
            RuntimeError: If timer was not started and never entered.
        """
        if self._elapsed is not None:
            return self._elapsed
        if self._start_time is not None:
            # Still running, return current elapsed time
            return time.monotonic() - self._start_time
        return 0.0

    @property
    def stats(self) -> TimerStats:
        """
        Get detailed timing statistics.

        Returns:
            TimerStats with detailed timing information.

        Raises:
            RuntimeError: If timer has not been started.
        """
        if self._start_time is None:
            raise RuntimeError("Timer has not been started")

        elapsed = self.elapsed
        return TimerStats(
            elapsed=elapsed,
            start_time=self._start_time,
            end_time=self._end_time,
        )

    @property
    def is_running(self) -> bool:
        """Check if the timer is currently running."""
        return self._running

    @property
    def is_started(self) -> bool:
        """Check if the timer has been started."""
        return self._start_time is not None

    def start(self) -> None:
        """
        Start the timer manually.

        Raises:
            RuntimeError: If timer is already running.
        """
        if self._running:
            raise RuntimeError("Timer is already running")
        self._start_time = time.monotonic()
        self._end_time = None
        self._elapsed = None
        self._running = True

    def stop(self) -> float:
        """
        Stop the timer manually and return elapsed time.

        Returns:
            Elapsed time in seconds.

        Raises:
            RuntimeError: If timer is not running.
        """
        if not self._running:
            raise RuntimeError("Timer is not running")
        self._end_time = time.monotonic()
        self._elapsed = self._end_time - self._start_time  # type: ignore
        self._running = False
        return self._elapsed

    def reset(self) -> None:
        """Reset the timer to initial state."""
        self._start_time = None
        self._end_time = None
        self._elapsed = None
        self._running = False

    async def __aenter__(self) -> "AsyncTimer":
        """Enter the context manager and start timing if auto_start is True."""
        if self._auto_start:
            self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the context manager and stop timing if running."""
        if self._running:
            self.stop()
        return False

    def __repr__(self) -> str:
        if self._start_time is None:
            return "AsyncTimer(not started)"
        if self._running:
            return f"AsyncTimer(running, elapsed={self.elapsed:.6f}s)"
        return f"AsyncTimer(elapsed={self._elapsed:.6f}s)"


@asynccontextmanager
async def async_timer() -> AsyncTimer:
    """
    Context manager that automatically starts timing.

    This is a convenience function that creates an AsyncTimer with
    auto_start=True, so timing begins immediately upon entering the context.

    Args:
        None

    Yields:
        AsyncTimer instance with timing started.

    Example:
        async with async_timer() as timer:
            await some_async_operation()

        print(f"Took {timer.elapsed:.3f} seconds")
    """
    timer_instance = AsyncTimer(auto_start=True)
    async with timer_instance:
        yield timer_instance


def timer(
    name: str | None = None,
    logger: Callable[[str], Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for measuring and logging async function execution time.

    Args:
        name: Optional name for the timed operation. Defaults to function name.
        logger: Optional callable for logging. If None, results are not logged.
                Common use: print, logging.info, or custom logger.

    Returns:
        Decorated async function that tracks execution time.

    Example:
        @timer()
        async def my_function():
            await asyncio.sleep(1)

        await my_function()  # Returns normally, timing attached as attribute

    Example with logging:
        import logging
        @timer(name="API Call", logger=logging.info)
        async def fetch_data(url):
            ...

    Example accessing elapsed time:
        @timer()
        async def process_data():
            ...

        result = await process_data()
        elapsed = process_data.elapsed  # Access timing info
    """
    import functools

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            timer_instance = AsyncTimer(auto_start=True)
            async with timer_instance:
                result = await func(*args, **kwargs)

            # Store elapsed time on the function for later access
            wrapper.elapsed = timer_instance.elapsed  # type: ignore

            # Log if logger provided
            if logger is not None:
                op_name = name or func.__name__
                logger(f"{op_name} completed in {timer_instance.elapsed:.6f}s")

            return result

        wrapper.elapsed = 0.0  # type: ignore
        return wrapper

    return decorator


__all__ = [
    "AsyncTimer",
    "TimerStats",
    "async_timer",
    "timer",
]
