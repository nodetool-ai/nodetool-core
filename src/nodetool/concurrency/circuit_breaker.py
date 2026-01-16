import asyncio
import enum
import time
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class CircuitState(enum.Enum):
    """Possible states for the circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when an operation is rejected due to an open circuit."""

    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message)


class AsyncCircuitBreaker:
    """
    An asynchronous circuit breaker for preventing cascading failures.

    The circuit breaker pattern protects against cascading failures by:
    - Tracking the failure count and success count
    - Opening (rejecting all requests) after reaching a failure threshold
    - Transitioning to half-open to test if the service has recovered
    - Closing (allowing requests) when enough consecutive successes occur

    Example:
        # Open after 5 failures, reset every 60 seconds, allow 3 successes to close
        breaker = AsyncCircuitBreaker(
            failure_threshold=5,
            recovery_time=60.0,
            success_threshold=3,
        )

        try:
            async with breaker:
                return await make_api_request()
        except CircuitBreakerError:
            return cached_response
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_time: float = 60.0,
        success_threshold: int = 3,
        monitored_exceptions: tuple[type[Exception], ...] = (Exception,),
    ):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening (default: 5).
            recovery_time: Seconds to wait before attempting recovery (default: 60.0).
            success_threshold: Consecutive successes needed in half-open to close (default: 3).
            monitored_exceptions: Exception types that count as failures (default: all exceptions).

        Raises:
            ValueError: If failure_threshold <= 0, recovery_time <= 0, or success_threshold <= 0.
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be a positive integer")
        if recovery_time <= 0:
            raise ValueError("recovery_time must be a positive number")
        if success_threshold <= 0:
            raise ValueError("success_threshold must be a positive integer")

        self._failure_threshold = failure_threshold
        self._recovery_time = recovery_time
        self._success_threshold = success_threshold
        self._monitored_exceptions = monitored_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Return the current state of the circuit breaker."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Return the number of consecutive failures."""
        return self._failure_count

    @property
    def failure_threshold(self) -> int:
        """Return the failure threshold that triggers opening."""
        return self._failure_threshold

    @property
    def recovery_time(self) -> float:
        """Return the recovery time in seconds."""
        return self._recovery_time

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return False
        return time.monotonic() - self._last_failure_time >= self._recovery_time

    async def _try_open(self) -> None:
        """Attempt to transition from open to half-open state."""
        async with self._lock:
            if self._state == CircuitState.OPEN and self._should_attempt_recovery():
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Async callable to execute.

        Returns:
            The return value of the function.

        Raises:
            CircuitBreakerError: If the circuit is open.
            The original exception if the function fails and circuit remains closed.
        """
        await self._try_open()

        if self._state == CircuitState.OPEN:
            raise CircuitBreakerError(f"Circuit breaker is open (will retry after {self._recovery_time}s)")

        try:
            result = await func()
            await self._on_success()
            return result
        except self._monitored_exceptions:
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        """Handle a successful operation."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _on_failure(self) -> None:
        """Handle a failed operation."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
            elif self._state == CircuitState.CLOSED and self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN

    async def __aenter__(self) -> "AsyncCircuitBreaker":
        await self._try_open()

        if self._state == CircuitState.OPEN:
            raise CircuitBreakerError(f"Circuit breaker is open (will retry after {self._recovery_time}s)")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            await self._on_success()
        else:
            should_handle = any(isinstance(exc_val, exc_type) for exc_type in self._monitored_exceptions)
            if should_handle:
                await self._on_failure()

    def reset(self) -> None:
        """
        Reset the circuit breaker to closed state with zeroed counters.

        Example:
            breaker.reset()
            # Circuit is now CLOSED with fresh counters
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None


__all__ = ["AsyncCircuitBreaker", "CircuitBreakerError", "CircuitState"]
