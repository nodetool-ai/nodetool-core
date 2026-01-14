import asyncio
import time
from enum import Enum
from types import TracebackType
from typing import Any, Callable, Self

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when the circuit breaker is open."""

    def __init__(self, message: str | None = None) -> None:
        self.message: str = message or "Circuit breaker is open"
        super().__init__(self.message)


class CircuitBreaker:
    """
    A circuit breaker implementation for preventing cascade failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests are rejected immediately
    - HALF_OPEN: Testing if the service has recovered

    Example:
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            success_threshold=3,
        )

        async with breaker:
            await make_external_api_call()

        # Or use as a decorator
        @breaker
        async def fetch_data(url):
            ...
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3,
        exception_type: type[Exception] = Exception,
    ) -> None:
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening the circuit.
            recovery_timeout: Seconds to wait before trying HALF_OPEN state.
            success_threshold: Successes needed in HALF_OPEN to close circuit.
            exception_type: Base exception type to count as failures.
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be a positive integer")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be a positive number")

        self.failure_threshold: int = failure_threshold
        self.recovery_timeout: float = recovery_timeout
        self.success_threshold: int = success_threshold
        self.exception_type: type[Exception] = exception_type

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: float | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Return the current state of the circuit breaker."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and self._should_attempt_reset():
                return CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        """Return the current failure count."""
        return self._failure_count

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        assert self._last_failure_time is not None
        return time.monotonic() - self._last_failure_time >= self.recovery_timeout

    async def _try_close(self) -> None:
        """Attempt to close the circuit after successful recovery."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    log.info("Circuit breaker closed")

    async def _trip(self) -> None:
        """Trip the circuit breaker to OPEN state."""
        self._state = CircuitState.OPEN
        self._last_failure_time = time.monotonic()
        self._failure_count = 0
        self._success_count = 0
        log.warning(f"Circuit breaker opened, will attempt recovery in {self.recovery_timeout}s")

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if exc_val is not None and isinstance(exc_val, self.exception_type):
            await self.record_failure(exc_val)
        else:
            await self.record_success()
        return False

    async def acquire(self) -> None:
        """Acquire permission to execute through the circuit breaker."""
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker is open, rejecting request. Will retry after {self.recovery_timeout}s"
                )

            if current_state == CircuitState.HALF_OPEN:
                log.debug("Circuit breaker in HALF_OPEN state, allowing test request")

    async def record_success(self) -> None:
        """Record a successful execution."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    log.info("Circuit breaker closed")
            else:
                self._failure_count = 0

    async def record_failure(self, exception: Exception) -> None:
        """Record a failed execution."""
        async with self._lock:
            self._failure_count += 1

            if self.state == CircuitState.HALF_OPEN:
                await self._trip()
                log.error(f"Circuit breaker tripped during HALF_OPEN: {exception}")
            elif self._failure_count >= self.failure_threshold:
                await self._trip()
                log.error(f"Circuit breaker tripped after {self.failure_threshold} failures: {exception}")

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Use as a decorator for async functions.

        Example:
            @CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
            async def fetch_data(url):
                ...
        """
        import functools

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self:
                return await func(*args, **kwargs)

        return wrapper


async def with_circuit_breaker(
    func: Callable[[], Any],
    breaker: CircuitBreaker,
) -> Any:
    """
    Execute a function through a circuit breaker.

    Args:
        func: Async callable to execute.
        breaker: CircuitBreaker instance to use.

    Returns:
        The result of the function.

    Raises:
        CircuitBreakerError: If the circuit is open.
        Any exception from the function (tracked for failure counting).

    Example:
        breaker = CircuitBreaker(failure_threshold=5)

        try:
            result = await with_circuit_breaker(
                lambda: fetch_api_data(),
                breaker,
            )
        except CircuitBreakerError:
            result = None  # Use cached/fallback value
    """
    async with breaker:
        return await func()


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitState",
    "with_circuit_breaker",
]
