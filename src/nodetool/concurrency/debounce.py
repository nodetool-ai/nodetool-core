import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar


class AsyncDebounceError(Exception):
    """Exception raised when debounce operation fails."""

    pass


T = TypeVar("T")


class AsyncDebounce(Generic[T]):
    """
    Debounce async function calls, delaying execution until a quiet period.

    When an async function is called, it schedules execution after a wait
    period. If called again before the period elapses, the timer resets.
    Only the last call within the quiet period executes.

    Args:
        func: The async function to debounce
        wait: Quiet period in seconds before execution (default: 0.3)

    Example:
        ```python
        debounced_save = AsyncDebounce(save_to_database, wait=1.0)

        # Multiple rapid calls only trigger one execution after 1 second
        await debounced_save(data)
        await debounced_save(data)
        await debounced_save(data)
        ```
    """

    _func: Callable[..., Awaitable[T]]
    _wait: float
    _lock: asyncio.Lock
    _args: tuple[Any, ...]
    _kwargs: dict[str, Any]
    _generation: int
    _active_gen: int | None

    def __init__(
        self,
        func: Callable[..., Awaitable[T]],
        wait: float = 0.3,
    ) -> None:
        self._func = func
        self._wait = wait
        self._lock = asyncio.Lock()
        self._args = ()
        self._kwargs = {}
        self._generation = 0
        self._active_gen = None

    async def __call__(self, *args: Any, **kwargs: Any) -> T | None:
        """
        Schedule the debounced function call.

        Args:
            *args: Positional arguments for the wrapped function
            **kwargs: Keyword arguments for the wrapped function

        Returns:
            The result of the wrapped function, or None if cancelled
        """
        async with self._lock:
            self._generation += 1
            current_gen = self._generation
            self._args = args
            self._kwargs = kwargs

        await asyncio.sleep(self._wait)

        async with self._lock:
            if self._active_gen is not None:
                return None
            if current_gen != self._generation:
                return None

            self._active_gen = current_gen
            call_args = self._args
            call_kwargs = self._kwargs

        try:
            return await self._func(*call_args, **call_kwargs)
        finally:
            async with self._lock:
                if self._active_gen == current_gen:
                    self._active_gen = None

    async def cancel(self) -> None:
        """
        Cancel any pending debounced execution.
        """
        async with self._lock:
            self._generation += 1


def debounce(
    wait: float = 0.3,
) -> Callable[[Callable[..., Awaitable[T]]], "AsyncDebounce[Any]"]:
    """
    Create a debounced version of an async function.

    Args:
        wait: Quiet period in seconds before execution (default: 0.3)

    Returns:
        Decorator that creates an AsyncDebounce wrapper

    Example:
        ```python
        @debounce(wait=0.5)
        async def process_item(item: Item) -> None:
            await expensive_operation(item)
        ```
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> AsyncDebounce[T]:
        return AsyncDebounce(func, wait)

    return decorator
