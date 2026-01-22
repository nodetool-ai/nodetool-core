import asyncio
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class DebounceError(Exception):
    """Raised when a debounced function is called during cooldown."""

    def __init__(self, message: str = "Function call rejected during debounce cooldown"):
        super().__init__(message)


class AsyncDebounce:
    """
    An asynchronous debounce utility that delays function execution until a quiet period.

    Debouncing ensures that a function is only called after a specified quiet period
    has elapsed since the last invocation. This is useful for:
    - Rate limiting API calls (e.g., search-as-you-type)
    - Batching rapid updates (e.g., UI state changes)
    - Aggregating log messages
    - Preventing duplicate submissions

    Example:
        # Create a debounced function that waits 300ms after the last call
        debounced_save = AsyncDebounce(wait_seconds=0.3)(save_to_database)

        # Rapid calls are accumulated, only the last one executes
        await debounced_save(data1)  # queued
        await debounced_save(data2)  # queued, replaces data1
        await debounced_save(data3)  # queued, replaces data2
        # After 300ms of silence, save_to_database(data3) is called

        # Control methods are available on the debounced function
        debounced_save.flush()  # Execute immediately
        debounced_save.cancel()  # Cancel pending execution

        # Use leading edge to execute immediately on first call
        debounced_search = AsyncDebounce(wait_seconds=0.3, leading=True)(search_api)

        await debounced_search("a")
        await debounced_search("ab")
        await debounced_search("abc")
        # search_api("a") executes immediately, other calls are debounced
    """

    def __init__(
        self,
        wait_seconds: float,
        leading: bool = False,
        trailing: bool = True,
        max_wait: float | None = None,
    ):
        """
        Initialize the debounce utility.

        Args:
            wait_seconds: Quiet period in seconds before executing.
            leading: Execute on the leading edge (first call) instead of trailing.
            trailing: Execute on the trailing edge (after quiet period).
            max_wait: Maximum time to wait before forcing execution (optional).

        Raises:
            ValueError: If wait_seconds <= 0 or conflicting edge settings.
        """
        if wait_seconds <= 0:
            raise ValueError("wait_seconds must be a positive number")

        if not leading and not trailing:
            raise ValueError("At least one of leading or trailing must be True")

        self._wait_seconds = wait_seconds
        self._leading = leading
        self._trailing = trailing
        self._max_wait = max_wait

        self._lock = asyncio.Lock()
        self._timer: asyncio.Task[Any] | None = None
        self._pending: dict[int, tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = {}
        self._call_count = 0
        self._execute_at: float | None = None

    @property
    def wait_seconds(self) -> float:
        """Return the quiet period in seconds."""
        return self._wait_seconds

    @property
    def is_pending(self) -> bool:
        """Return True if a call is scheduled but not yet executed."""
        return self._timer is not None and not self._timer.done()

    @property
    def pending_count(self) -> int:
        """Return the number of pending calls."""
        return len(self._pending)

    @property
    def call_count(self) -> int:
        """Return the number of calls that have been debounced."""
        return self._call_count

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorate an async function to make it debounced.

        Args:
            func: The async function to debounce.

        Returns:
            A debounced version of the function with control methods.
        """
        func_id = id(func)

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self._execute(func_id, func, args, kwargs)

        wrapper._is_debounced = True  # type: ignore[attr-defined]
        wrapper.flush = self.flush  # type: ignore[attr-defined]
        wrapper.cancel = self.cancel  # type: ignore[attr-defined]
        wrapper.reset = self.reset  # type: ignore[attr-defined]
        wrapper.call_count = property(lambda self: self._call_count)  # type: ignore[attr-defined]
        wrapper.wait_seconds = property(lambda self: self._wait_seconds)  # type: ignore[attr-defined]
        wrapper.is_pending = property(lambda self: self.is_pending)  # type: ignore[attr-defined]

        return wrapper

    def _create_timer(
        self,
        func_id: int,
        func: Callable[..., Any],
    ) -> asyncio.Task[Any]:
        """Create and start the debounce timer."""
        wait_time = self._wait_seconds

        if self._max_wait is not None and self._execute_at is not None:
            now = time.monotonic()
            remaining = self._execute_at - now + self._wait_seconds
            if remaining > self._max_wait:
                wait_time = self._max_wait - (now - (self._execute_at or now))

        async def run_timer() -> None:
            await asyncio.sleep(wait_time)
            async with self._lock:
                if self._timer is None:
                    return

                self._timer = None
                if func_id in self._pending:
                    func, args, kwargs = self._pending.pop(func_id)
                    self._call_count += 1
                    try:
                        await func(*args, **kwargs)
                    except Exception:
                        raise

        return asyncio.create_task(run_timer())

    async def _execute(
        self,
        func_id: int,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Execute a debounced function call.

        Args:
            func_id: Unique identifier for the function.
            func: The function to execute.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            The return value of the function, or None for trailing edge.
        """
        async with self._lock:
            self._pending[func_id] = (func, args, kwargs)
            self._call_count += 1
            now = time.monotonic()

            if self._max_wait is not None:
                if self._execute_at is None:
                    self._execute_at = now + self._max_wait
                if now >= self._execute_at:
                    del self._pending[func_id]
                    self._call_count -= 1
                    return await func(*args, **kwargs)

            if self._leading and self._timer is None:
                del self._pending[func_id]
                result = await func(*args, **kwargs)
                if self._trailing:
                    self._timer = self._create_timer(func_id, func)
                return result

            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

            if self._trailing:
                self._timer = self._create_timer(func_id, func)

            return None

    async def flush(self) -> list[Any]:
        """
        Immediately execute all pending calls.

        Returns:
            List of return values from executed functions.

        Example:
            debounced = AsyncDebounce(wait_seconds=0.5)(save_data)
            await debounced(data1)
            await debounced(data2)
            results = await debounced.flush()  # Executes immediately
        """
        async with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

            pending = dict(self._pending)
            self._pending.clear()
            self._execute_at = None

        results = []
        for func, args, kwargs in pending.values():
            try:
                result = await func(*args, **kwargs)
                results.append(result)
            except Exception:
                raise

        return results

    def cancel(self) -> None:
        """
        Cancel any pending execution.

        Example:
            debounced = AsyncDebounce(wait_seconds=0.5)(save_data)
            await debounced(data)
            debounced.cancel()  # The save is cancelled
        """
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._pending.clear()
        self._execute_at = None

    def reset(self) -> None:
        """Reset the debounce state, cancelling any pending execution."""
        self.cancel()
        self._call_count = 0


class DebounceGroup:
    """
    A group of debounced functions that share a timer.

    All functions in the group are debounced together - when any function is called,
    all pending executions are flushed and a new quiet period begins.

    Example:
        # Group related API calls that should be debounced together
        api_group = DebounceGroup(wait_seconds=0.5)

        search = api_group(search_api)
        save = api_group(save_api)

        await search(query1)  # Triggers quiet period
        await search(query2)  # Quiet period restarts
        await save(data1)     # Both are flushed, quiet period restarts

        # Control methods are available on wrapped functions
        search.flush()
        save.cancel()
    """

    def __init__(self, wait_seconds: float):
        """
        Initialize the debounce group.

        Args:
            wait_seconds: Quiet period in seconds before executing.
        """
        self._wait_seconds = wait_seconds
        self._lock = asyncio.Lock()
        self._timer: asyncio.Task[Any] | None = None
        self._pending: dict[int, tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = {}
        self._next_id = 0

    @property
    def wait_seconds(self) -> float:
        """Return the quiet period in seconds."""
        return self._wait_seconds

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorate a function to be part of this debounce group.

        Args:
            func: The async function to wrap.

        Returns:
            A wrapped function that participates in the group debounce.
        """
        func_id = self._next_id
        self._next_id += 1

        async def wrapper(*args: Any, **kwargs: Any) -> None:
            async with self._lock:
                self._pending[func_id] = (func, args, kwargs)

                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None

                async def run_group() -> None:
                    async with self._lock:
                        pending = dict(self._pending)
                        self._pending = {}
                        self._timer = None

                    for func, args, kwargs in pending.values():
                        await func(*args, **kwargs)

                self._timer = asyncio.create_task(run_group())

            return None

        wrapper._is_grouped = True  # type: ignore[attr-defined]
        wrapper.flush = self.flush  # type: ignore[attr-defined]
        wrapper.cancel = self.cancel  # type: ignore[attr-defined]

        return wrapper

    async def flush(self) -> None:
        """Flush all pending calls immediately."""
        async with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

            pending = dict(self._pending)
            self._pending = {}

        for func, args, kwargs in pending.values():
            await func(*args, **kwargs)

    def cancel(self) -> None:
        """Cancel all pending calls."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._pending.clear()


__all__ = ["AsyncDebounce", "DebounceError", "DebounceGroup"]
