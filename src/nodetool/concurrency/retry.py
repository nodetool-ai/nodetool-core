import asyncio
import random
from typing import Any, Coroutine, TypeVar
from collections.abc import Callable

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

T = TypeVar("T")


async def retry_with_exponential_backoff(
    func: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """
    Retry an async function with exponential backoff and optional jitter.

    This utility provides a robust pattern for handling transient failures
    by automatically retrying failed operations with increasing delays.

    Args:
        func: Async function to execute and retry on failure.
        max_retries: Maximum number of retry attempts (default: 3).
                     Use -1 for unlimited retries.
        initial_delay: Initial delay in seconds between retries (default: 1.0).
        max_delay: Maximum delay cap in seconds (default: 60.0).
        exponential_base: Base for exponential backoff calculation (default: 2.0).
                          Delay = initial_delay * (exponential_base ** attempt)
        jitter: Whether to add random jitter to prevent thundering herd (default: True).
        retryable_exceptions: Tuple of exception types to retry on (default: all exceptions).

    Returns:
        The return value of the successfully executed function.

    Raises:
        The last exception if all retries are exhausted.

    Example:
        async def fetch_data(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()

        data = await retry_with_exponential_backoff(
            lambda: fetch_data("https://api.example.com/data"),
            max_retries=5,
            initial_delay=0.5,
            retryable_exceptions=(aiohttp.ClientError,),
        )
    """
    if max_retries < -1:
        raise ValueError("max_retries must be -1 (unlimited) or >= 0")

    delay = initial_delay
    attempt = 0

    while True:
        try:
            return await func()
        except retryable_exceptions as e:
            if max_retries != -1 and attempt >= max_retries:
                log.error(
                    f"Operation failed after {max_retries} retries: {e}",
                    extra={"attempt": attempt + 1, "max_retries": max_retries},
                )
                raise

            log.warning(
                f"Operation failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {e}",
                extra={"attempt": attempt + 1, "next_delay": delay},
            )

            actual_delay = delay * random.uniform(0.5, 1.5) if jitter else delay

            await asyncio.sleep(actual_delay)

            delay = min(delay * exponential_base, max_delay)
            attempt += 1


class RetryPolicy:
    """
    A configurable retry policy for fine-grained control over retry behavior.

    This class allows defining complex retry strategies with different settings
    for different exception types and conditions.

    Example:
        # Retry on rate limit with longer delays
        policy = RetryPolicy(
            max_retries=5,
            initial_delay=2.0,
            retryable_exceptions=(aiohttp.ClientError,),
        )

        # Custom predicate for determining if an exception is retryable
        policy = RetryPolicy(
            max_retries=3,
            retryable_predicate=lambda e: isinstance(e, ValueError) and "retry" in str(e),
        )
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
        retryable_predicate: Callable[[Exception], bool] | None = None,
    ):
        """
        Initialize a retry policy.

        Args:
            max_retries: Maximum number of retry attempts (-1 for unlimited).
            initial_delay: Initial delay in seconds.
            max_delay: Maximum delay cap.
            exponential_base: Base for exponential backoff.
            jitter: Whether to add random jitter.
            retryable_exceptions: Exception types that are retryable.
            retryable_predicate: Custom function to determine if an exception is retryable.
        """
        self.max_retries: int = max_retries
        self.initial_delay: float = initial_delay
        self.max_delay: float = max_delay
        self.exponential_base: float = exponential_base
        self.jitter: bool = jitter
        self.retryable_exceptions: tuple[type[Exception], ...] = retryable_exceptions
        self.retryable_predicate: Callable[[Exception], bool] | None = retryable_predicate

    async def execute(self, func: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Execute a function with this retry policy."""
        return await retry_with_exponential_backoff(
            func=func,
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            max_delay=self.max_delay,
            exponential_base=self.exponential_base,
            jitter=self.jitter,
            retryable_exceptions=self.retryable_exceptions,
        )

    def __call__(self, func: Callable[[], Coroutine[Any, Any, T]]) -> Callable[[], Coroutine[Any, Any, T]]:
        """
        Use as a decorator for async functions.

        Example:
            @RetryPolicy(max_retries=3, initial_delay=0.5)
            async def fetch_data(url):
                ...
        """
        import functools

        @functools.wraps(func)
        async def wrapper() -> T:
            return await self.execute(func)

        return wrapper


__all__ = ["RetryPolicy", "retry_with_exponential_backoff"]
