import asyncio
from typing import Any, Callable


class DebouncedFunction:
    """
    A wrapper around a debounced async function that provides control methods.

    This class is returned by debounce() when decorating a function.
    It provides flush(), cancel(), and pending properties for controlling
    the debounced execution.

    Example:
        debounced_save = debounce(0.5)(save_to_database)
        await debounced_save(data)  # Will execute after 0.5s if no new calls
        debounced_save.flush()  # Execute immediately
    """

    def __init__(
        self,
        func: Callable[..., Any],
        wait_seconds: float,
    ):
        """
        Initialize the debounced function wrapper.

        Args:
            func: The async function to debounce.
            wait_seconds: The debounce delay in seconds.
        """
        self._func = func
        self._wait_seconds = wait_seconds
        self._timer: asyncio.Task[Any] | None = None
        self._lock = asyncio.Lock()
        self._args: tuple[Any, ...] | None = None
        self._kwargs: dict[str, Any] | None = None

    async def _execute(self) -> Any:
        """Wait for the debounce period and then execute the function."""
        await asyncio.sleep(self._wait_seconds)
        async with self._lock:
            args = self._args
            kwargs = self._kwargs or {}
            self._args = None
            self._kwargs = None
            self._timer = None
        if args is not None:
            return await self._func(*args, **kwargs)
        return None

    async def __call__(self, *args: Any, **kwargs: Any) -> asyncio.Task[Any]:
        """
        Schedule the function to be called after the debounce period.

        Unlike typical debounce implementations, this method returns a Task
        that can be awaited to get the result. The task is started immediately
        and runs in the background.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            A task representing the pending or completed execution.
        """
        async with self._lock:
            self._args = args
            self._kwargs = kwargs
            if self._timer is not None and not self._timer.done():
                self._timer.cancel()
            self._timer = asyncio.create_task(self._execute())
        return self._timer

    def flush(self) -> asyncio.Task[Any] | None:
        """
        Cancel any pending debounce and schedule immediate execution.

        Returns:
            The task representing the immediate execution, or None if no pending call.
        """
        if self._timer is not None and not self._timer.done():
            self._timer.cancel()

        async def execute_immediately() -> Any:
            async with self._lock:
                args = self._args
                kwargs = self._kwargs or {}
                self._args = None
                self._kwargs = None
                self._timer = None
            if args is not None:
                return await self._func(*args, **kwargs)
            return None

        return asyncio.create_task(execute_immediately())

    def cancel(self) -> None:
        """
        Cancel any pending debounced call without executing it.
        """
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._args = None
        self._kwargs = None

    @property
    def pending(self) -> bool:
        """Return True if there's a pending debounced call that hasn't executed yet."""
        return self._timer is not None and not self._timer.done()

    @property
    def wait_seconds(self) -> float:
        """Return the debounce delay in seconds."""
        return self._wait_seconds


def debounce(wait_seconds: float) -> Callable[[Callable[..., Any]], DebouncedFunction]:
    """
    Create a debounce decorator for async functions.

    This creates a DebouncedFunction wrapper that debounces calls and provides
    control methods like flush(), cancel(), and pending.

    Args:
        wait_seconds (float): The debounce delay in seconds.

    Returns:
        A function that accepts an async function and returns a DebouncedFunction.

    Example:
        @debounce(0.5)
        async def save_document(content: str) -> None:
            await save_to_db(content)

        # Calls within 0.5s will be debounced
        debounced_save = save_document
        await debounced_save("first")
        await debounced_save("second")  # first is cancelled

        # Control methods
        debounced_save.cancel()  # Cancel pending call
        debounced_save.flush()  # Execute immediately
    """

    def decorator(func: Callable[..., Any]) -> DebouncedFunction:
        return DebouncedFunction(func, wait_seconds)

    return decorator


class AsyncDebounce:
    """
    An async debounce utility for delaying function execution until a wait period
    has passed without new calls. This is useful for rate-limiting bursts of events
    like user input, API calls, or network requests.

    Unlike synchronous debounce implementations, this supports async functions and
    provides proper cancellation handling.

    This class is primarily designed to be used as a decorator with arguments:

        @AsyncDebounce(wait_seconds=0.5)
        async def save_to_database(data: dict) -> None:
            await database.save(data)

    For direct usage with a function, use the `debounce` function instead:

        debounced_save = debounce(0.5)(save_to_database)
    """

    def __init__(
        self,
        *,
        wait_seconds: float,
    ):
        """
        Initialize the debounce decorator.

        Args:
            wait_seconds: The debounce delay in seconds. Must be positive.

        Raises:
            ValueError: If wait_seconds is not positive.
        """
        if wait_seconds <= 0:
            raise ValueError("wait_seconds must be a positive number")
        self._wait_seconds = wait_seconds

    def __call__(self, func: Callable[..., Any]) -> DebouncedFunction:
        """
        Decorate a function with debounce behavior.

        Args:
            func: The async function to decorate.

        Returns:
            A DebouncedFunction wrapper that debounces calls and provides control methods.
        """
        return DebouncedFunction(func, self._wait_seconds)
