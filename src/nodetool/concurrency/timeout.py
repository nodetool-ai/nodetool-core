import asyncio
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_ASYNCIO_TIMEOUT_ERROR = asyncio.TimeoutError


class TimeoutError(Exception):
    """Raised when an operation times out."""

    def __init__(self, timeout_seconds: float, message: str | None = None):
        self.timeout_seconds = timeout_seconds
        self.message = message or f"Operation timed out after {timeout_seconds}s"
        super().__init__(self.message)


def timeout(
    seconds: float,
    exception_message: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that applies a timeout to an async function.

    Args:
        seconds: Timeout in seconds.
        exception_message: Custom error message (optional).

    Returns:
        Decorated function that raises TimeoutError on timeout.

    Example:
        @timeout(5.0)
        async def fetch_data(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()

        try:
            data = await fetch_data("https://api.example.com/data")
        except TimeoutError:
            data = None
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except _ASYNCIO_TIMEOUT_ERROR:
                raise TimeoutError(seconds, exception_message) from None

        return wrapper

    return decorator


async def with_timeout(
    coro: Callable[..., Any],
    timeout_seconds: float,
    timeout_exception: type[Exception] = TimeoutError,
    exception_message: str | None = None,
) -> Any:
    """
    Wrap an async coroutine with a timeout.

    Args:
        coro: Async callable to execute.
        timeout_seconds: Timeout in seconds.
        timeout_exception: Exception type to raise on timeout (default: TimeoutError).
        exception_message: Custom error message (optional).

    Returns:
        The result of the coroutine.

    Raises:
        timeout_exception: If the operation times out.

    Example:
        async def fetch_data(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()

        try:
            data = await with_timeout(
                lambda: fetch_data("https://api.example.com/data"),
                timeout_seconds=5.0,
            )
        except TimeoutError:
            data = None
    """
    try:
        return await asyncio.wait_for(coro(), timeout=timeout_seconds)
    except _ASYNCIO_TIMEOUT_ERROR:
        raise timeout_exception(
            timeout_seconds,
            exception_message or f"Operation timed out after {timeout_seconds}s",
        ) from None


class TimeoutPolicy:
    """
    A configurable timeout policy for fine-grained control over timeout behavior.

    Example:
        # Short timeout for non-critical operations
        policy = TimeoutPolicy(default_timeout=2.0)

        # Longer timeout with custom exception
        class ApiTimeoutError(TimeoutError):
            pass

        policy = TimeoutPolicy(
            default_timeout=30.0,
            timeout_exception=ApiTimeoutError,
        )

        # Use as context manager
        async with policy.timeout(10.0):
            await fetch_api_data()
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        timeout_exception: type[Exception] = TimeoutError,
        default_message: str | None = None,
    ):
        """
        Initialize a timeout policy.

        Args:
            default_timeout: Default timeout in seconds (default: 30.0).
            timeout_exception: Exception type to raise on timeout.
            default_message: Default error message for timeouts.
        """
        self.default_timeout = default_timeout
        self.timeout_exception = timeout_exception
        self.default_message = default_message

    async def execute(
        self,
        coro: Callable[..., Any],
        timeout_seconds: float | None = None,
        exception_message: str | None = None,
    ) -> Any:
        """
        Execute a coroutine with this timeout policy.

        Args:
            coro: Async callable to execute.
            timeout_seconds: Override timeout (uses default if None).
            exception_message: Custom error message (uses default if None).

        Returns:
            The result of the coroutine.

        Raises:
            self.timeout_exception: If the operation times out.
        """
        timeout_sec = timeout_seconds if timeout_seconds is not None else self.default_timeout
        msg = exception_message or self.default_message

        try:
            return await asyncio.wait_for(coro(), timeout=timeout_sec)
        except _ASYNCIO_TIMEOUT_ERROR:
            raise self.timeout_exception(
                timeout_sec,
                msg or f"Operation timed out after {timeout_sec}s",
            ) from None

    def timeout(
        self,
        timeout_seconds: float,
        exception_message: str | None = None,
    ) -> "TimeoutContext":
        """
        Create a context manager for timeout control.

        Args:
            timeout_seconds: Timeout in seconds.
            exception_message: Custom error message (optional).

        Returns:
            TimeoutContext manager.

        Example:
            async with policy.timeout(10.0):
                await some_operation()
        """
        return TimeoutContext(
            timeout_seconds=timeout_seconds,
            timeout_exception=self.timeout_exception,
            exception_message=exception_message,
        )

    def __call__(
        self,
        func: Callable[..., Any],
    ) -> Callable[..., Any]:
        """
        Use as a decorator with the default timeout.

        Args:
            func: Async function to decorate.

        Returns:
            Decorated function with timeout.

        Example:
            @TimeoutPolicy(default_timeout=5.0)
            async def fetch_data(url):
                ...
        """
        import functools

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.execute(
                lambda: func(*args, **kwargs),
                timeout_seconds=self.default_timeout,
            )

        return wrapper


class TimeoutContext:
    """
    Context manager for applying timeouts to async operations.

    Example:
        async with TimeoutContext(timeout_seconds=5.0):
            await some_long_running_operation()
    """

    def __init__(
        self,
        timeout_seconds: float,
        timeout_exception: type[Exception] = TimeoutError,
        exception_message: str | None = None,
    ):
        """
        Initialize the timeout context.

        Args:
            timeout_seconds: Timeout in seconds.
            timeout_exception: Exception type to raise on timeout.
            exception_message: Custom error message.
        """
        self.timeout_seconds = timeout_seconds
        self.timeout_exception = timeout_exception
        self.exception_message = exception_message
        self._task: asyncio.Task[Any] | None = None

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False

    async def run(self, coro: Callable[..., Any]) -> Any:
        """
        Run a coroutine with the configured timeout.

        Args:
            coro: Async callable to execute.

        Returns:
            The result of the coroutine.

        Raises:
            self.timeout_exception: If the operation times out.
        """
        try:
            return await asyncio.wait_for(coro(), timeout=self.timeout_seconds)
        except _ASYNCIO_TIMEOUT_ERROR:
            raise self.timeout_exception(
                self.timeout_seconds,
                self.exception_message or f"Operation timed out after {self.timeout_seconds}s",
            ) from None


__all__ = [
    "TimeoutContext",
    "TimeoutError",
    "TimeoutPolicy",
    "timeout",
    "with_timeout",
]
