"""
Async debounce and throttle utilities for controlling execution frequency.

These utilities help manage the rate at which functions are executed,
which is useful for rate limiting, preventing excessive API calls, and
reducing computational load.
"""

import asyncio
from collections.abc import Callable
from typing import Any, Coroutine, TypeVar

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class AsyncDebounce:
    """
    Debounce async function calls - delay execution until after a pause.

    Debouncing ensures that a function is only executed after a specified
    delay has passed since the last invocation. Useful for search-as-you-type,
    auto-save, and other scenarios where you want to wait for user input to pause.

    Example:
        debounce = AsyncDebounce(delay=0.5)

        async def search(query):
            return await api.search(query)

        # Create task for debounced execution (non-blocking)
        task = asyncio.create_task(debounce.execute(lambda: search("hello")))
        task = asyncio.create_task(debounce.execute(lambda: search("hello world")))
        # Only search("hello world") will run after 0.5s delay
        await task  # Wait for completion if needed

        # Cancel pending execution
        await debounce.cancel()
    """

    def __init__(self, delay: float):
        """
        Initialize a debouncer.

        Args:
            delay: Delay in seconds to wait after the last call before executing.
        """
        if delay <= 0:
            raise ValueError("delay must be positive")

        self._delay: float = delay
        self._timer_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current_func: Any = None
        self._completion_events: list[asyncio.Event] = []

    async def execute(self, func: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """
        Execute a function with debouncing.

        If called multiple times within the delay period, only the last
        function will be executed after the delay.

        Note: This method is designed to be called with create_task() for
        non-blocking behavior. If awaited directly, it will wait for the
        debounce delay.

        Args:
            func: Async function to execute.

        Returns:
            The result of the executed function.
        """
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop

        # Create an event for this call
        event = asyncio.Event()

        async with self._lock:
            # Store the function to execute
            self._current_func = func
            self._completion_events.append(event)

            # If no timer is running, start one
            if self._timer_task is None or self._timer_task.done():
                self._timer_task = asyncio.create_task(self._timer_loop())

        # Wait for completion
        await event.wait()

        # Return the result or re-raise the exception
        if hasattr(self, "_last_exception") and self._last_exception is not None:
            exc = self._last_exception
            self._last_exception = None
            raise exc
        if hasattr(self, "_last_result"):
            return self._last_result  # type: ignore
        else:
            raise RuntimeError("Execution completed without result")

    async def _timer_loop(self) -> None:
        """Internal timer loop that handles delayed execution."""
        try:
            while True:
                await asyncio.sleep(self._delay)

                async with self._lock:
                    if not self._completion_events:
                        # No pending calls, exit the loop
                        break

                    # Get the function to execute
                    func_to_execute = self._current_func
                    events_to_notify = self._completion_events.copy()
                    self._completion_events.clear()

                should_exit = False
                if func_to_execute is not None:
                    # Execute the function
                    try:
                        self._last_result = await func_to_execute()
                        self._last_exception = None
                    except Exception as e:
                        # Store the actual exception so waiters can re-raise it
                        self._last_exception = e
                    finally:
                        # Notify all waiters
                        for event in events_to_notify:
                            event.set()

                        # Check if new calls came in during execution
                        async with self._lock:
                            if not self._completion_events:
                                # No new calls, exit
                                should_exit = True

                if should_exit:
                    break
        except asyncio.CancelledError:
            # Notify any waiting events before exiting
            async with self._lock:
                for event in self._completion_events:
                    event.set()
                self._completion_events.clear()

    async def cancel(self) -> bool:
        """
        Cancel any pending execution.

        Returns:
            True if a task was cancelled, False if no task was pending.
        """
        async with self._lock:
            if self._timer_task is not None and not self._timer_task.done():
                self._timer_task.cancel()
                self._timer_task = None

            if self._completion_events:
                # Notify all waiters that we're cancelling
                for event in self._completion_events:
                    event.set()
                self._completion_events.clear()
                self._current_func = None
                return True

            return False

    def is_pending(self) -> bool:
        """
        Check if there is a pending execution.

        Returns:
            True if a task is pending, False otherwise.
        """
        return bool(self._completion_events) or (
            self._timer_task is not None and not self._timer_task.done()
        )


class AsyncThrottle:
    """
    Throttle async function calls - limit execution rate.

    Throttling ensures that a function is executed at most once per
    specified time interval. Useful for rate limiting API calls,
    preventing excessive resource usage, and controlling execution frequency.

    Example:
        throttle = AsyncThrottle(interval=1.0)

        async def fetch_data():
            return await api.fetch()

        # Will execute immediately, then subsequent calls within 1s
        # will be throttled and not executed
        await throttle.execute(lambda: fetch_data())  # Executes
        await throttle.execute(lambda: fetch_data())  # Throttled (skipped)

        # Wait for interval to pass
        await asyncio.sleep(1.0)
        await throttle.execute(lambda: fetch_data())  # Executes again

        # Reset the throttle timer
        await throttle.reset()
    """

    def __init__(self, interval: float):
        """
        Initialize a throttle.

        Args:
            interval: Minimum time in seconds between executions.
        """
        if interval <= 0:
            raise ValueError("interval must be positive")

        self._interval: float = interval
        self._last_execution: float = 0.0
        self._lock = asyncio.Lock()

    async def execute(
        self, func: Callable[[], Coroutine[Any, Any, T]], skip_if_throttled: bool = True
    ) -> T | None:
        """
        Execute a function with throttling.

        Args:
            func: Async function to execute.
            skip_if_throttled: If True, return None when throttled.
                             If False, wait until throttle allows execution.

        Returns:
            The result of the executed function, or None if skipped/throttled.
        """
        async with self._lock:
            now = asyncio.get_running_loop().time()
            time_since_last = now - self._last_execution

            if time_since_last >= self._interval:
                # Execute immediately
                self._last_execution = now
                return await func()
            elif skip_if_throttled:
                # Skip execution
                log.debug(
                    "Throttled function call skipped",
                    extra={
                        "time_since_last": time_since_last,
                        "interval": self._interval,
                    },
                )
                return None
            else:
                # Wait for remaining time
                pass

        # Wait outside the lock
        await asyncio.sleep(self._interval - time_since_last)

        # Re-acquire lock and execute
        async with self._lock:
            self._last_execution = asyncio.get_running_loop().time()
            return await func()

    async def reset(self) -> None:
        """Reset the throttle timer, allowing immediate next execution."""
        async with self._lock:
            self._last_execution = 0.0

    def can_execute(self) -> bool:
        """
        Check if execution is currently allowed (not throttled).

        Returns:
            True if execution would be allowed now, False if throttled.
        """
        now = asyncio.get_running_loop().time()
        time_since_last = now - self._last_execution
        return time_since_last >= self._interval


class AdaptiveThrottle:
    """
    Adaptive throttle that adjusts interval based on success/failure rates.

    This throttle automatically increases the interval when failures occur
    and decreases it when operations succeed, providing automatic rate
    adaptation for unreliable services.

    Example:
        throttle = AdaptiveThrottle(
            min_interval=0.1,
            max_interval=10.0,
            backoff_multiplier=2.0,
            recovery_multiplier=0.8,
        )

        async def unreliable_api_call():
            return await api.call()

        result = await throttle.execute(lambda: unreliable_api_call())
        # Interval automatically adjusts based on success/failure
    """

    def __init__(
        self,
        min_interval: float = 0.1,
        max_interval: float = 60.0,
        initial_interval: float = 1.0,
        backoff_multiplier: float = 2.0,
        recovery_multiplier: float = 0.8,
    ):
        """
        Initialize an adaptive throttle.

        Args:
            min_interval: Minimum interval in seconds.
            max_interval: Maximum interval in seconds.
            initial_interval: Starting interval in seconds.
            backoff_multiplier: Multiplier for interval on failure.
            recovery_multiplier: Multiplier for interval on success.
        """
        if min_interval <= 0 or max_interval <= 0 or initial_interval <= 0:
            raise ValueError("intervals must be positive")
        if min_interval > max_interval:
            raise ValueError("min_interval must be <= max_interval")
        if initial_interval < min_interval or initial_interval > max_interval:
            raise ValueError("initial_interval must be between min_interval and max_interval")

        self._min_interval = min_interval
        self._max_interval = max_interval
        self._initial_interval = initial_interval
        self._current_interval = initial_interval
        self._backoff_multiplier = backoff_multiplier
        self._recovery_multiplier = recovery_multiplier
        self._last_execution: float = 0.0
        self._lock = asyncio.Lock()

    async def execute(self, func: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """
        Execute a function with adaptive throttling.

        The interval automatically adjusts based on success/failure.

        Args:
            func: Async function to execute.

        Returns:
            The result of the executed function.
        """
        # Wait for throttle interval
        async with self._lock:
            now = asyncio.get_running_loop().time()
            time_since_last = now - self._last_execution

            if time_since_last < self._current_interval:
                # Release lock while waiting
                pass

        if time_since_last < self._current_interval:
            await asyncio.sleep(self._current_interval - time_since_last)

        # Execute function
        try:
            result = await func()
            # Success - reduce interval
            async with self._lock:
                self._last_execution = asyncio.get_running_loop().time()
                new_interval = self._current_interval * self._recovery_multiplier
                self._current_interval = max(new_interval, self._min_interval)
                log.debug(
                    "Adaptive throttle: success, reducing interval",
                    extra={
                        "old_interval": self._current_interval / self._recovery_multiplier,
                        "new_interval": self._current_interval,
                    },
                )
            return result
        except Exception as e:
            # Failure - increase interval
            async with self._lock:
                self._last_execution = asyncio.get_running_loop().time()
                new_interval = self._current_interval * self._backoff_multiplier
                self._current_interval = min(new_interval, self._max_interval)
                log.warning(
                    f"Adaptive throttle: failure, increasing interval: {e}",
                    extra={
                        "old_interval": self._current_interval / self._backoff_multiplier,
                        "new_interval": self._current_interval,
                    },
                )
            raise

    def get_current_interval(self) -> float:
        """Get the current throttle interval."""
        return self._current_interval

    async def reset(self) -> None:
        """Reset the throttle to initial interval."""
        async with self._lock:
            self._current_interval = self._initial_interval
            self._last_execution = 0.0


__all__ = ["AdaptiveThrottle", "AsyncDebounce", "AsyncThrottle"]
