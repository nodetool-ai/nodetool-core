import asyncio
from enum import Enum
from typing import Any, Callable, TypeVar

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when the circuit breaker is open and rejects an operation."""

    def __init__(self, message: str | None = None):
        self.message = message or "Circuit breaker is open, operation rejected"
        super().__init__(self.message)


async def _to_async(func: Callable[..., Any]) -> Any:
    """Convert a callable to awaitable if needed."""
    result = func()
    if asyncio.iscoroutine(result):
        return await result
    if asyncio.isfuture(result):
        return await result
    return result


class CircuitBreaker:
    """
    A circuit breaker implementation for protecting against cascading failures.

    The circuit breaker prevents an application from repeatedly trying to execute
    an operation that's likely to fail, allowing it to fail fast and give the
    downstream service time to recover.

    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Failure threshold reached, requests are rejected immediately
        - HALF_OPEN: Testing if service recovered, limited requests allowed

    Example:
        cb = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            success_threshold=3
        )

        async with cb:
            await make_external_api_call()

        # Or use as a decorator
        @CircuitBreaker(failure_threshold=3)
        async def fragile_operation():
            ...
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3,
        expected_exception: type[Exception] = Exception,
    ):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening the circuit.
            recovery_timeout: Seconds to wait before attempting recovery.
            success_threshold: Successes needed in HALF_OPEN to close circuit.
            expected_exception: Exception type that triggers failure counting.
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        if success_threshold <= 0:
            raise ValueError("success_threshold must be positive")

        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._expected_exception = expected_exception

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
        """Return the current failure count."""
        return self._failure_count

    @property
    def is_closed(self) -> bool:
        """Return True if the circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Return True if the circuit is open (rejecting requests)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Return True if the circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def _transition_to_open(self) -> None:
        """Transition to OPEN state and record failure time."""
        if self._state != CircuitState.OPEN:
            log.warning(
                f"Circuit breaker opening after {self._failure_count} failures",
                extra={"failure_count": self._failure_count, "threshold": self._failure_threshold},
            )
            self._state = CircuitState.OPEN
            self._last_failure_time = asyncio.get_event_loop().time()

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state for recovery testing."""
        if self._state == CircuitState.OPEN:
            log.info("Circuit breaker entering half-open state for recovery test")
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state (normal operation restored)."""
        if self._state != CircuitState.CLOSED:
            log.info("Circuit breaker closing - service recovered")
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    async def _check_recovery(self) -> None:
        """Check if enough time has passed to attempt recovery."""
        if self._state == CircuitState.OPEN and self._last_failure_time is not None:
            elapsed = asyncio.get_event_loop().time() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._transition_to_half_open()

    async def _record_success(self) -> None:
        """Record a successful execution."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _record_failure(self) -> None:
        """Record a failed execution."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._failure_threshold:
                    self._transition_to_open()

    async def execute(self, func: Callable[..., Any]) -> Any:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async callable or sync callable to execute.

        Returns:
            The return value of the function.

        Raises:
            CircuitBreakerError: If the circuit is open.
            func's original exception: If the function fails.
        """
        await self._check_recovery()

        if self._state == CircuitState.OPEN:
            raise CircuitBreakerError(f"Circuit open for {self._recovery_timeout}s, try again later")

        try:
            result = await _to_async(func)
            await self._record_success()
            return result
        except self._expected_exception:
            await self._record_failure()
            raise

    async def __aenter__(self) -> "CircuitBreaker":
        await self._check_recovery()
        if self._state == CircuitState.OPEN:
            raise CircuitBreakerError(f"Circuit open for {self._recovery_timeout}s, try again later")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is None:
            await self._record_success()
        else:
            await self._record_failure()
        return False

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Use as a decorator for async functions.

        Example:
            @CircuitBreaker(failure_threshold=3)
            async def fragile_service():
                ...
        """
        import functools

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.execute(lambda: func(*args, **kwargs))

        return wrapper

    def reset(self) -> None:
        """Reset the circuit breaker to closed state with zero failures."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None


__all__ = ["CircuitBreaker", "CircuitBreakerError", "CircuitState"]
