import asyncio
from typing import Any, Callable, Coroutine


class AsyncDebounce:
    """
    An async debounce decorator that delays execution until a quiet period.

    Debouncing ensures that a function is only called after a specified quiet
    period where no new calls are made. This is useful for:
    - Search-as-you-type (wait until user stops typing)
    - Auto-save (wait until user stops editing)
    - API request coalescing (combine multiple rapid requests into one)

    Example:
        async def save_document(content: str):
            await save_to_database(content)

        debounced_save = AsyncDebounce(save_document, wait=0.5)

        # Multiple rapid calls will be debounced into a single call
        await debounced_save("first")
        await debounced_save("second")
        await debounced_save("third")
        # Only one call to save_document("third") will execute after 0.5s
    """

    def __init__(
        self,
        func: Callable[..., Any],
        wait: float,
    ):
        """
        Initialize the debounce decorator.

        Args:
            func: The async function to debounce.
            wait: Quiet period in seconds before executing the function.
                  Must be non-negative.

        Raises:
            ValueError: If wait is negative or func is not callable.
        """
        if wait < 0:
            raise ValueError("wait must be a non-negative number")
        if not callable(func):
            raise ValueError("func must be callable")

        self._func = func
        self._wait = wait
        self._timer: asyncio.Task[None] | None = None
        self._args: tuple[Any, ...] = ()
        self._kwargs: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, None]:
        """
        Queue a call to the wrapped function.

        If a call is already queued, it will be replaced with the new call.
        The function will execute after the quiet period expires.

        Args:
            *args: Positional arguments for the wrapped function.
            **kwargs: Keyword arguments for the wrapped function.

        Returns:
            A coroutine that completes when the call is queued.
        """
        return self._queue_call(*args, **kwargs)

    async def _queue_call(self, *args: Any, **kwargs: Any) -> None:
        """Queue the call and schedule execution."""
        async with self._lock:
            self._args = args
            self._kwargs = kwargs

            if self._timer is not None:
                self._timer.cancel()

            self._timer = asyncio.create_task(self._wait_and_execute())

    async def _wait_and_execute(self) -> None:
        """Wait for the quiet period then execute the function."""
        await asyncio.sleep(self._wait)

        async with self._lock:
            if self._timer is not None:
                self._timer = None

            args = self._args
            kwargs = self._kwargs

        await self._func(*args, **kwargs)

    def flush(self) -> asyncio.Task[None] | None:
        """
        Immediately execute any pending call.

        This is useful when you need to ensure pending work completes
        before shutting down.

        Returns:
            A task representing the pending execution, or None if no pending call.
        """
        if self._timer is None or self._timer.done():
            return None

        self._timer.cancel()
        return asyncio.create_task(self._wait_and_execute())

    def cancel(self) -> None:
        """
        Cancel any pending execution.

        After calling cancel, any queued call will be discarded.
        """
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    @property
    def pending(self) -> bool:
        """Return True if a call is pending execution."""
        return self._timer is not None and not self._timer.done()


def async_debounce(
    func: Callable[..., Any] | None = None,
    wait: float = 0.3,
) -> Callable[[Callable[..., Any]], AsyncDebounce] | AsyncDebounce:
    """
    Decorator or function wrapper for async debouncing.

    Can be used as a decorator:

        @async_debounce(wait=0.5)
        async def search(query: str):
            results = await perform_search(query)
            return results

    Or as a function wrapper:

        async def save(data):
            await database.save(data)

        debounced_save = async_debounce(save, wait=1.0)

    Args:
        func: The async function to debounce (when used as decorator).
        wait: Quiet period in seconds before executing the function.

    Returns:
        A callable that wraps the function with debounce behavior.

    Raises:
        ValueError: If wait is negative.
    """
    if wait < 0:
        raise ValueError("wait must be a non-negative number")

    def decorator(func: Callable[..., Any]) -> AsyncDebounce:
        return AsyncDebounce(func, wait)

    if func is not None:
        return decorator(func)

    return decorator


__all__ = ["AsyncDebounce", "async_debounce"]
