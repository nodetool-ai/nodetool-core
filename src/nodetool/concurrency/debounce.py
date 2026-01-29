import asyncio
from collections.abc import Callable
from typing import Any, Coroutine, TypeVar, cast

T = TypeVar("T")


class DebouncedFunc:
    """
    An async function wrapper that implements debouncing behavior.

    Debouncing ensures that a function is only called after a quiet period
    where no new calls are made. Each new call resets the wait timer.
    This is useful for rate limiting, preventing duplicate requests,
    and handling rapid UI events.

    Example:
        # Create a debounced function with 500ms cooldown
        debounced_save = DebouncedFunc(save_to_database, wait=0.5)

        # Call it multiple times rapidly - only the last call executes
        await debounced_save({"data": "value1"})
        await debounced_save({"data": "value2"})  # Resets timer
        await debounced_save({"data": "value3"})  # Resets timer

        # After 500ms without new calls, the last value is processed
        # with: save_to_database({"data": "value3"})
    """

    def __init__(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        wait: float = 0.3,
    ):
        """
        Initialize the debounced function wrapper.

        Args:
            func: The async function to wrap.
            wait: Minimum time in seconds between executions (default: 0.3).

        Raises:
            ValueError: If wait <= 0.
        """
        if wait <= 0:
            raise ValueError("wait must be a positive number")

        self._func = func
        self._wait = wait
        self._lock = asyncio.Lock()
        self._pending: asyncio.Task[Any] | None = None
        self._exec_id: int = 0
        self._latest_args: tuple[Any, ...] | None = None
        self._latest_kwargs: dict[str, Any] | None = None

    @property
    def wait(self) -> float:
        """Return the debounce wait time in seconds."""
        return self._wait

    @property
    def pending(self) -> bool:
        """Return whether a call is currently pending execution."""
        return self._pending is not None and not self._pending.done()

    async def _run(self, exec_id: int) -> T | None:
        """Execute the function after waiting."""
        await asyncio.sleep(self._wait)

        async with self._lock:
            if self._exec_id != exec_id:
                return None
            args = self._latest_args
            kwargs = self._latest_kwargs
            self._latest_args = None
            self._latest_kwargs = None

        if args is None and kwargs is None:
            return None

        result = await self._func(*(args or ()), **(kwargs or {}))
        return cast("T | None", result)

    async def __call__(self, *args: Any, **kwargs: Any) -> None:
        """
        Call the debounced function.

        This schedules the function to be called after the wait period.
        If called again before the wait period expires, the timer resets.

        Args:
            *args: Positional arguments for the wrapped function.
            **kwargs: Keyword arguments for the wrapped function.
        """
        async with self._lock:
            self._exec_id += 1
            current_id = self._exec_id
            self._latest_args = args
            self._latest_kwargs = kwargs

            if self._pending is not None and not self._pending.done():
                self._pending.cancel()

            self._pending = asyncio.create_task(self._run(current_id))

    async def flush(self) -> T | None:
        """
        Immediately execute any pending call.

        This is useful when you need to ensure all pending calls
        are processed before continuing.

        Returns:
            The result of the wrapped function, or None if no pending call.
        """
        async with self._lock:
            if self._pending is None:
                return None
            pending = self._pending
            self._pending = None

        return await pending

    def cancel(self) -> None:
        """
        Cancel any pending execution.

        After calling cancel, any pending call is discarded and
        the next call will start fresh.
        """
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()
        self._pending = None
        self._latest_args = None
        self._latest_kwargs = None


def debounce(
    wait: float = 0.3,
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], DebouncedFunc]:
    """
    Decorator that debounces an async function.

    Debouncing ensures that a function is only called after a quiet period
    where no new calls are made. Each new call resets the wait timer.

    Args:
        wait: Minimum time in seconds between executions (default: 0.3).

    Returns:
        A decorator that wraps the function with debounce behavior.

    Raises:
        ValueError: If wait <= 0.

    Example:
        @debounce(wait=0.5)
        async def save_data(data: dict) -> None:
            await database.save(data)

        # Multiple calls within 500ms only trigger one execution
        await save_data({"key": "value1"})
        await save_data({"key": "value2"})
        await save_data({"key": "value3"})
        # Only the last call with value3 is executed
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> DebouncedFunc:
        return DebouncedFunc(func=func, wait=wait)

    return decorator


__all__ = ["DebouncedFunc", "debounce"]
