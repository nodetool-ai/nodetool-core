import asyncio
import time
from typing import Any, Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")


def debounce(
    func: Callable[..., Awaitable[T]],
    wait: float,
    *,
    leading: bool = False,
    trailing: bool = True,
    max_wait: float | None = None,
) -> Callable[..., Awaitable[T]]:
    """
    Create a debounced version of an async function.

    Debouncing ensures that a function is only called after a period of inactivity.
    This is useful for rate-limiting function calls in response to rapid events,
    such as user input, network messages, or sensor readings.

    Args:
        func: The async function to debounce.
        wait: Minimum time in seconds between function calls.
        leading: If True, call the function on the leading edge of the wait period.
                 Defaults to False (function is called after wait period).
        trailing: If True, call the function on the trailing edge of the wait period.
                  Defaults to True.
        max_wait: Optional maximum time in seconds before forcing a call.
                  If set, the function will be called at least once even if
                  continuous calls are made. Defaults to None (no max wait).

    Returns:
        A debounced version of the input function.

    Example:
        # Basic usage - call after 300ms of inactivity
        debounced_save = debounce(save_data, wait=0.3)

        # Call immediately on first invocation, then debounce subsequent calls
        debounced_save = debounce(save_data, wait=0.3, leading=True)

        # Call on both leading and trailing edges
        debounced_save = debounce(save_data, wait=0.3, leading=True, trailing=True)

        # Force call after 5 seconds even if calls keep coming
        debounced_search = debounce(search, wait=0.3, max_wait=5.0)

        # Usage
        for i in range(10):
            debounced_save(data)  # Calls are debounced
            await asyncio.sleep(0.05)
    """
    if wait <= 0:
        raise ValueError("wait must be a positive number")
    if max_wait is not None and max_wait <= 0:
        raise ValueError("max_wait must be a positive number")
    if not leading and not trailing:
        raise ValueError("At least one of leading or trailing must be True")

    lock = asyncio.Lock()
    scheduled_time: float = 0.0
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}

    async def wrapper(*call_args: Any, **call_kwargs: Any) -> T:
        nonlocal scheduled_time, args, kwargs

        args = call_args
        kwargs = call_kwargs
        now = time.monotonic()

        async with lock:
            if leading and scheduled_time == 0:
                scheduled_time = now
                return await func(*call_args, **call_kwargs)

            should_execute = False
            if max_wait is not None and scheduled_time > 0:
                elapsed = now - (scheduled_time - wait)
                if elapsed >= max_wait:
                    should_execute = True

            if should_execute:
                scheduled_time = now
                return await func(*call_args, **call_kwargs)

            scheduled_time = now + wait

        await asyncio.sleep(wait)

        async with lock:
            elapsed = time.monotonic() - (scheduled_time - wait)
            if elapsed >= wait:
                scheduled_time = 0.0
                return await func(*call_args, **call_kwargs)

            scheduled_time = 0.0

        if trailing:
            return await func(*call_args, **call_kwargs)

        return await func(*args, **kwargs)

    return wrapper


class DebouncedCall(Generic[T]):
    """
    A context manager for managing debounced async function calls.

    This class provides fine-grained control over debounced execution,
    allowing explicit flush and cancellation operations.

    Example:
        debounced = DebouncedCall(save_data, wait=0.3)

        # Use as context manager - function called on exit
        async with debounced:
            await save_data(part1)
            await save_data(part2)

        # Manual control with flush
        await debounced.trigger()
        await debounced.flush()

        # Cancel pending call
        await debounced.cancel()
    """

    def __init__(
        self,
        func: Callable[..., Awaitable[T]],
        wait: float,
        *,
        leading: bool = False,
        trailing: bool = True,
    ):
        """
        Initialize the debounced call manager.

        Args:
            func: The async function to debounce.
            wait: Minimum time in seconds between function calls.
            leading: If True, call the function on the leading edge.
            trailing: If True, call the function on the trailing edge.

        Raises:
            ValueError: If wait <= 0 or both leading and trailing are False.
        """
        if wait <= 0:
            raise ValueError("wait must be a positive number")
        if not leading and not trailing:
            raise ValueError("At least one of leading or trailing must be True")

        self._func = func
        self._wait = wait
        self._leading = leading
        self._trailing = trailing
        self._lock = asyncio.Lock()
        self._scheduled_time: float = 0.0
        self._args: tuple[Any, ...] = ()
        self._kwargs: dict[str, Any] = {}

    async def trigger(self, *args: Any, **kwargs: Any) -> T | None:
        """
        Trigger a debounced call with the given arguments.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """
        async with self._lock:
            self._args = args
            self._kwargs = kwargs

            if self._leading and self._scheduled_time == 0:
                self._scheduled_time = time.monotonic()
                await self._func(*args, **kwargs)
                return

            self._scheduled_time = time.monotonic() + self._wait

        if self._trailing:
            await asyncio.sleep(self._wait)

            async with self._lock:
                elapsed = time.monotonic() - (self._scheduled_time - self._wait)
                if elapsed >= self._wait:
                    self._scheduled_time = 0.0
                    return await self._func(*args, **kwargs)

        self._scheduled_time = 0.0
        await self._func(*args, **kwargs)

    async def __aenter__(self) -> "DebouncedCall[T]":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.flush()

    async def flush(self) -> T | None:
        """
        Force execution of any pending debounced call.

        Returns:
            The result of the function call, or None if no call was pending.
        """
        async with self._lock:
            scheduled_time = self._scheduled_time
            self._scheduled_time = 0.0

        if scheduled_time > 0:
            return await self._func(*self._args, **self._kwargs)
        return None

    async def cancel(self) -> None:
        """Cancel any pending debounced call."""
        async with self._lock:
            self._scheduled_time = 0.0

    @property
    def pending(self) -> bool:
        """Return True if a call is pending."""
        return self._scheduled_time > 0

    @property
    def wait_time(self) -> float:
        """Return the debounce wait time in seconds."""
        return self._wait


__all__ = ["DebouncedCall", "debounce"]
