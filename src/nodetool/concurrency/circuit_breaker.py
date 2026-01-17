import asyncio
import enum
import time
from dataclasses import dataclass
from typing import Any, Callable

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class CircuitState(enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: float | None = None
    last_failure_exception: Exception | None = None

    def __post_init__(self) -> None:
        if isinstance(self.last_failure_time, (int, float)):
            self.last_failure_time = float(self.last_failure_time)

    def copy(self) -> "CircuitBreakerStats":
        """Create a copy of the stats."""
        return CircuitBreakerStats(
            total_calls=self.total_calls,
            successful_calls=self.successful_calls,
            failed_calls=self.failed_calls,
            rejected_calls=self.rejected_calls,
            last_failure_time=self.last_failure_time,
            last_failure_exception=self.last_failure_exception,
        )


class CircuitBreakerError(Exception):
    """Raised when a circuit breaker is open and rejects calls."""

    def __init__(self, message: str, state: CircuitState | None = None):
        super().__init__(message)
        self.state = state


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for a circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening the circuit.
        success_threshold: Number of successes in half-open state to close.
        timeout_seconds: Time in seconds before attempting to half-open.
        half_open_max_calls: Maximum calls allowed in half-open state.
        exception_filter: Callable to determine if an exception should count
            as a failure. Return False to ignore the exception.
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    exception_filter: Callable[[Exception], bool] | None = None


class CircuitBreaker:
    """
    A circuit breaker implementation for preventing cascading failures.

    The circuit breaker operates in three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is down, requests fail immediately
    - HALF_OPEN: Testing recovery, limited requests allowed

    This pattern is essential for maintaining system resilience when
    interacting with external services that may become temporarily unavailable.

    Example:
        async def call_external_api():
            ...

        # Create a circuit breaker that opens after 5 failures
        breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=30.0
        )

        # Use as a context manager
        async with breaker:
            result = await call_external_api()

        # Or use the call method directly
        result = await breaker.call(call_external_api)

        # Access statistics
        stats = breaker.stats
        if stats.failed_calls > 0:
            log.warning(f" Circuit breaker triggered: {stats.failed_calls} failures")
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        """
        Initialize the circuit breaker.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()
        self._stats = CircuitBreakerStats()

        if self._config.failure_threshold == 0:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            log.debug("Circuit breaker initialized in OPEN state (failure_threshold=0)")

    @property
    def config(self) -> CircuitBreakerConfig:
        """Get the circuit breaker configuration."""
        return self._config

    @property
    def state(self) -> CircuitState:
        """Get the current circuit state."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.timeout_seconds:
                return CircuitState.HALF_OPEN
        return self._state

    def _update_state_for_timeout(self) -> None:
        """Update state if timeout has expired."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                self._half_open_calls = 0
                log.debug("Circuit breaker timeout expired, transitioning to half-open")

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self._stats

    def _should_count_failure(self, exception: Exception) -> bool:
        """Determine if an exception should count as a failure."""
        if self._config.exception_filter is None:
            return True
        return self._config.exception_filter(exception)

    async def _on_failure(self, exception: Exception) -> None:
        """Handle a failed call."""
        self._stats.failed_calls += 1
        self._stats.last_failure_time = time.monotonic()
        self._stats.last_failure_exception = exception

        if self._state == CircuitState.HALF_OPEN:
            await self._open()
            return

        self._failure_count += 1
        log.warning(
            f"Circuit breaker failure ({self._failure_count}/{self._config.failure_threshold}): {exception}",
            extra={"failure_count": self._failure_count, "threshold": self._config.failure_threshold},
        )

        if self._failure_count >= self._config.failure_threshold:
            await self._open()

    async def _on_success(self) -> None:
        """Handle a successful call."""
        self._stats.successful_calls += 1

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            log.debug(
                f"Circuit breaker success in half-open state ({self._success_count}/{self._config.success_threshold})",
                extra={"success_count": self._success_count, "threshold": self._config.success_threshold},
            )

            if self._success_count >= self._config.success_threshold:
                await self._close()

        self._failure_count = 0

    async def _open(self) -> None:
        """Transition to open state."""
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._half_open_calls = 0
        self._success_count = 0
        log.error(
            f"Circuit breaker opened after {self._failure_count} failures",
            extra={"failure_count": self._failure_count, "timeout_seconds": self._config.timeout_seconds},
        )

    async def _close(self) -> None:
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._opened_at = None
        log.info("Circuit breaker closed - service recovered")

    async def _can_proceed(self) -> tuple[bool, CircuitState]:
        """
        Check if a call can proceed and update state if needed.

        Returns:
            Tuple of (can_proceed, current_state)
        """
        async with self._lock:
            self._update_state_for_timeout()
            current_state = self._state

            if current_state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                log.warning(
                    "Circuit breaker is open - call rejected",
                    extra={"opened_at": self._opened_at, "timeout_seconds": self._config.timeout_seconds},
                )
                return False, current_state

            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._config.half_open_max_calls:
                    self._stats.rejected_calls += 1
                    log.warning(
                        "Circuit breaker half-open limit reached - call rejected",
                        extra={"half_open_calls": self._half_open_calls, "max_calls": self._config.half_open_max_calls},
                    )
                    return False, current_state
                self._half_open_calls += 1

            return True, current_state

    async def call(self, func: Callable[[], Any]) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Async function to execute.

        Returns:
            The return value of the function.

        Raises:
            CircuitBreakerError: If the circuit is open.
            Exception: Any exception from the wrapped function.
        """
        self._stats.total_calls += 1

        can_proceed, state = await self._can_proceed()
        if not can_proceed:
            raise CircuitBreakerError(
                f"Circuit breaker is open - request rejected in state {state.value}",
                state=state,
            )

        try:
            result = await func()
            await self._on_success()
            return result
        except Exception as e:
            if self._should_count_failure(e):
                await self._on_failure(e)
            raise

    async def __aenter__(self) -> "CircuitBreaker":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        if exc_type is not None and exc_val is not None:
            if self._should_count_failure(exc_val):
                await self._on_failure(exc_val)

    def __repr__(self) -> str:
        """String representation of the circuit breaker."""
        return (
            f"CircuitBreaker(state={self.state.value}, failures={self._failure_count}, successes={self._success_count})"
        )


class MultiCircuitBreaker:
    """
    A circuit breaker that manages separate breakers for different keys.

    Useful when you need separate circuit breakers for different services,
    endpoints, or resource types.

    Example:
        # Create a multi-breaker for different API endpoints
        breaker = MultiCircuitBreaker(failure_threshold=3, timeout_seconds=30.0)

        # Call with different service keys
        result = await breaker.call("user_service", fetch_users)
        result = await breaker.call("order_service", fetch_orders)
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        default_config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize the multi-circuit breaker.

        Args:
            config: Default config for all breakers (shared reference).
            default_config: Config to copy for new breakers.
        """
        self._default_config = config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def _get_breaker(self, key: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a key."""
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(
                config=CircuitBreakerConfig(
                    failure_threshold=self._default_config.failure_threshold,
                    success_threshold=self._default_config.success_threshold,
                    timeout_seconds=self._default_config.timeout_seconds,
                    half_open_max_calls=self._default_config.half_open_max_calls,
                    exception_filter=self._default_config.exception_filter,
                )
            )
        return self._breakers[key]

    async def call(self, key: str, func: Callable[[], Any]) -> Any:
        """
        Execute a function through a keyed circuit breaker.

        Args:
            key: Unique identifier for the circuit (e.g., service name, URL).
            func: Async function to execute.

        Returns:
            The return value of the function.
        """
        async with self._lock:
            breaker = self._get_breaker(key)

        return await breaker.call(func)

    def get_stats(self, key: str) -> CircuitBreakerStats | None:
        """Get statistics for a specific circuit breaker."""
        breaker = self._breakers.get(key)
        return breaker.stats if breaker else None

    def get_all_stats(self) -> dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers."""
        return {key: breaker.stats.copy() for key, breaker in self._breakers.items()}

    def reset(self, key: str | None = None) -> None:
        """Reset circuit breaker(s)."""
        if key is not None:
            self._breakers.pop(key, None)
        else:
            self._breakers.clear()

    def __repr__(self) -> str:
        """String representation of the multi-circuit breaker."""
        return f"MultiCircuitBreaker(keys={len(self._breakers)})"


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerStats",
    "CircuitState",
    "MultiCircuitBreaker",
]
