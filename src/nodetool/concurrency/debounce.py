import asyncio
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


async def async_debounce(
    func: Callable[[], Any],
    wait: float,
    max_wait: float | None = None,
) -> Any:
    """
    Debounce an async function - only execute after wait seconds have passed
    since the last call.

    This utility is useful for rate-limiting operations that should only execute
    after a period of inactivity, such as saving data after user input stops.

    Args:
        func: Async function to debounce.
        wait: Minimum time in seconds between executions.
        max_wait: Optional maximum time in seconds - forces execution if reached.

    Returns:
        The return value of the function, or None if still waiting.

    Example:
        # Debounce a save function that only runs 500ms after the last call
        save = lambda: save_to_database(data)

        # Call multiple times - only the last one executes
        save()
        save()  # Reset timer
        save()  # Reset timer
        # After 500ms without calls, save_to_database is executed
    """
    raise NotImplementedError("async_debounce needs implementation")


class DebouncePolicy:
    """
    A configurable debounce policy for fine-grained control over debounce behavior.

    Debouncing delays function execution until a specified quiet period has elapsed.
    This is useful for batching rapid events, reducing API calls, and preventing
    resource thrashing.

    Example:
        # Debounce with 500ms wait, 5s maximum wait
        policy = DebouncePolicy(wait=0.5, max_wait=5.0)

        async with policy:
            await save_user_input(data)

        # Or use as a decorator
        @DebouncePolicy(wait=0.3)
        async def process_update(update):
            await database.update(update)
    """

    def __init__(
        self,
        wait: float = 0.3,
        max_wait: float | None = None,
    ):
        """
        Initialize the debounce policy.

        Args:
            wait: Minimum time in seconds between executions.
            max_wait: Optional maximum time - forces execution if reached.

        Raises:
            ValueError: If wait <= 0.
        """
        if wait <= 0:
            raise ValueError("wait must be a positive number")

        self._wait = wait
        self._max_wait = max_wait
        self._lock = asyncio.Lock()
        self._timer_task: asyncio.Task[None] | None = None
        self._pending_func: Callable[[], Any] | None = None
        self._executed: asyncio.Event = asyncio.Event()
        self._cancelled: bool = False
        self._execution_time: float = 0.0
        self._max_deadline: float = 0.0
        self._executed_func: bool = False

    @property
    def wait(self) -> float:
        """Return the debounce wait time in seconds."""
        return self._wait

    @property
    def max_wait(self) -> float | None:
        """Return the maximum wait time, or None if not set."""
        return self._max_wait

    async def __aenter__(self) -> "DebouncePolicy":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.flush()

    async def schedule(self, func: Callable[[], Any]) -> None:
        """
        Schedule a function to be executed after the debounce period.

        If called again before the period elapses, the timer resets.
        If max_wait is set and exceeded, execution happens immediately.

        Args:
            func: Async function to schedule.
        """
        async with self._lock:
            self._pending_func = func
            self._cancelled = False
            self._executed.clear()
            self._executed_func = False

            now = time.monotonic()

            if self._max_wait is not None:
                if self._max_deadline == 0:
                    self._max_deadline = now + self._max_wait
                self._execution_time = min(now + self._wait, self._max_deadline)
            else:
                self._execution_time = now + self._wait

            if self._timer_task is not None and not self._timer_task.done():
                try:
                    self._timer_task.cancel()
                except Exception:
                    pass
                self._timer_task = None

            self._timer_task = asyncio.create_task(self._run_timer())

    async def flush(self) -> bool:
        """
        Force immediate execution of any pending function.

        Returns:
            True if a pending function was executed, False if none was pending.
        """
        async with self._lock:
            if self._timer_task is not None and not self._timer_task.done():
                self._timer_task.cancel()
                self._timer_task = None

            if self._pending_func is not None and not self._cancelled:
                func = self._pending_func
                self._pending_func = None
                self._cancelled = True
                self._execution_time = 0.0
                self._max_deadline = 0.0
                await func()
                self._executed.set()
                return True

            return False

    async def _run_timer(self) -> None:
        """Internal timer task that manages debounce timing."""
        try:
            async with self._lock:
                if self._cancelled or self._pending_func is None or self._executed_func:
                    return
                execution_time = self._execution_time

            now = time.monotonic()
            sleep_time = max(0, execution_time - now)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            async with self._lock:
                if self._cancelled or self._pending_func is None or self._executed_func:
                    return

                if self._execution_time != execution_time:
                    return

                now = time.monotonic()
                if now < execution_time:
                    return

                self._executed_func = True
                func = self._pending_func
                self._pending_func = None
                self._execution_time = 0.0
                self._max_deadline = 0.0
                self._executed.set()

            await func()
        except asyncio.CancelledError:
            pass

    async def wait_executed(self) -> None:
        """Wait until the scheduled function has been executed."""
        await self._executed.wait()


__all__ = ["DebouncePolicy", "async_debounce"]
