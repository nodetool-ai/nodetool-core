import asyncio
from enum import Enum
from typing import Optional


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when a circuit breaker is open and requests are blocked."""

    def __init__(self, message: str = "Circuit breaker is open") -> None:
        self.message = message
        super().__init__(self.message)


class CircuitBreaker:
    """
    An async circuit breaker implementation for preventing cascading failures.

    The circuit breaker protects external service calls by monitoring failures
    and temporarily blocking requests when failure thresholds are exceeded.
    This prevents cascading failures and allows services time to recover.

    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Too many failures, requests are blocked immediately
        - HALF_OPEN: Testing if service has recovered, limited requests allowed

    Example:
        cb = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            success_threshold=3
        )

        async with cb:
            await make_external_api_call()

        try:
            async with cb:
                await make_external_api_call()
        except CircuitBreakerError:
            print("Service is unavailable, circuit is open")
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold (int): Number of failures before opening the circuit.
                                    Must be greater than 0. Default: 5.
            recovery_timeout (float): Time in seconds before attempting recovery.
                                     Must be positive. Default: 30.0.
            success_threshold (int): Number of successes in HALF_OPEN state
                                    before closing the circuit. Must be > 0. Default: 3.
            name (Optional[str]): Optional name for identification in logs.

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be a positive integer")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be a positive number")
        if success_threshold <= 0:
            raise ValueError("success_threshold must be a positive integer")

        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> Optional[str]:
        """Return the optional name of the circuit breaker."""
        return self._name

    @property
    def state(self) -> CircuitState:
        """Return the current state of the circuit breaker."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Return the current failure count."""
        return self._failure_count

    @property
    def failure_threshold(self) -> int:
        """Return the failure threshold that triggers circuit opening."""
        return self._failure_threshold

    @property
    def recovery_timeout(self) -> float:
        """Return the timeout in seconds before attempting recovery."""
        return self._recovery_timeout

    @property
    def success_threshold(self) -> int:
        """Return the success threshold needed to close the circuit from half_open."""
        return self._success_threshold

    @property
    def last_failure_time(self) -> Optional[float]:
        """Return the timestamp of the last failure, or None if no failures."""
        return self._last_failure_time

    def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._failure_count += 1
        self._last_failure_time = asyncio.get_event_loop().time()

        if self._state == CircuitState.CLOSED and self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._success_count = 0

    def _record_success(self) -> None:
        """Record a success and potentially close the circuit."""
        self._failure_count = 0
        self._last_failure_time = None

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._success_threshold:
                self._state = CircuitState.CLOSED
                self._success_count = 0

    def _try_open_from_timeout(self) -> bool:
        """
        Check if circuit should transition from OPEN to HALF_OPEN based on timeout.

        Returns:
            bool: True if transition occurred, False otherwise.
        """
        if self._state == CircuitState.OPEN and self._last_failure_time is not None:
            current_time = asyncio.get_event_loop().time()
            if current_time - self._last_failure_time >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                return True
        return False

    async def call(
        self,
        coro,
        timeout: Optional[float] = None,
    ):
        """
        Execute a coroutine through the circuit breaker.

        Args:
            coro: The coroutine or awaitable to execute.
            timeout (Optional[float]): Optional timeout for the coroutine execution.

        Returns:
            The result of the coroutine.

        Raises:
            CircuitBreakerError: If the circuit is open.
            asyncio.TimeoutError: If the timeout expires.
            Any exception raised by the coroutine is propagated.
        """
        async with self._lock:
            self._try_open_from_timeout()

            if self._state == CircuitState.OPEN:
                raise CircuitBreakerError(f"Circuit breaker is open. Last failure: {self._last_failure_time}")

        try:
            if timeout is not None:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro

            async with self._lock:
                self._record_success()
            return result

        except Exception:
            async with self._lock:
                self._record_failure()
            raise

    async def __aenter__(self) -> "CircuitBreaker":
        """Enter the async context manager."""
        async with self._lock:
            self._try_open_from_timeout()

            if self._state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self._name}' is open" if self._name else "Circuit breaker is open"
                )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager."""
        if exc_type is None:
            async with self._lock:
                self._record_success()
        else:
            async with self._lock:
                self._record_failure()

    def reset(self) -> None:
        """
        Reset the circuit breaker to its initial closed state.

        This clears all failure counts and resets the state to CLOSED.
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
