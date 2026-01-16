import asyncio
import time

import pytest

from nodetool.concurrency.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)


class TestCircuitBreakerInitialization:
    """Tests for CircuitBreaker initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        cb = CircuitBreaker()
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 30.0
        assert cb.success_threshold == 3
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        cb = CircuitBreaker(
            failure_threshold=10,
            recovery_timeout=60.0,
            success_threshold=5,
            name="test_cb",
        )
        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 60.0
        assert cb.success_threshold == 5
        assert cb.name == "test_cb"
        assert cb.state == CircuitState.CLOSED

    def test_init_invalid_failure_threshold(self):
        """Test that invalid failure_threshold raises ValueError."""
        with pytest.raises(ValueError, match="failure_threshold must be a positive integer"):
            CircuitBreaker(failure_threshold=0)

        with pytest.raises(ValueError, match="failure_threshold must be a positive integer"):
            CircuitBreaker(failure_threshold=-1)

    def test_init_invalid_recovery_timeout(self):
        """Test that invalid recovery_timeout raises ValueError."""
        with pytest.raises(ValueError, match="recovery_timeout must be a positive number"):
            CircuitBreaker(recovery_timeout=0)

        with pytest.raises(ValueError, match="recovery_timeout must be a positive number"):
            CircuitBreaker(recovery_timeout=-1)

    def test_init_invalid_success_threshold(self):
        """Test that invalid success_threshold raises ValueError."""
        with pytest.raises(ValueError, match="success_threshold must be a positive integer"):
            CircuitBreaker(success_threshold=0)

        with pytest.raises(ValueError, match="success_threshold must be a positive integer"):
            CircuitBreaker(success_threshold=-1)


class TestCircuitBreakerBasicOperation:
    """Tests for basic circuit breaker operation."""

    @pytest.mark.asyncio
    async def test_successful_call_closes_circuit(self):
        """Test that successful calls keep the circuit closed."""
        cb = CircuitBreaker(failure_threshold=3)

        async def success():
            return "success"

        result = await cb.call(success())
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_failed_call_increases_failure_count(self):
        """Test that failed calls increase the failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        async def failure():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await cb.call(failure())

        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        """Test that circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        async def always_fail():
            raise ValueError("test error")

        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(always_fail())

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self):
        """Test that open circuit rejects calls with CircuitBreakerError."""
        cb = CircuitBreaker(failure_threshold=2)

        async def always_fail():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await cb.call(always_fail())

        with pytest.raises(ValueError):
            await cb.call(always_fail())

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError):
            await cb.call(asyncio.sleep(0))

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self):
        """Test that circuit transitions to half_open after recovery timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        async def always_fail():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await cb.call(always_fail())

        with pytest.raises(ValueError):
            await cb.call(always_fail())

        assert cb.state == CircuitState.OPEN

        await asyncio.sleep(0.15)

        async def do_nothing():
            return None

        await cb.call(do_nothing())

        assert cb.state == CircuitState.HALF_OPEN


class TestCircuitBreakerContextManager:
    """Tests for circuit breaker async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_success(self):
        """Test context manager with successful operation."""
        cb = CircuitBreaker(failure_threshold=3)

        async with cb:
            result = await asyncio.sleep(0, result="done")

        assert result == "done"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_failure(self):
        """Test context manager with failed operation."""
        cb = CircuitBreaker(failure_threshold=3)

        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("test error")

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1

    @pytest.mark.asyncio
    async def test_context_manager_opens_circuit(self):
        """Test that context manager opens circuit after threshold."""
        cb = CircuitBreaker(failure_threshold=2)

        for _ in range(2):
            with pytest.raises(ValueError):
                async with cb:
                    raise ValueError("test error")

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_context_manager_rejects_when_open(self):
        """Test that context manager rejects calls when circuit is open."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)

        async def fail_once():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            async with cb:
                await fail_once()

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError):
            async with cb:
                await asyncio.sleep(0)


class TestCircuitBreakerRecovery:
    """Tests for circuit breaker recovery behavior."""

    @pytest.mark.asyncio
    async def test_successful_calls_in_half_open_closes_circuit(self):
        """Test that successful calls in half_open close the circuit."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
            success_threshold=2,
        )

        async def always_fail():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await cb.call(always_fail())

        assert cb.state == CircuitState.OPEN

        await asyncio.sleep(0.1)

        await cb.call(asyncio.sleep(0))
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.failure_count == 0

        await cb.call(asyncio.sleep(0))
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failed_call_in_half_open_reopens_circuit(self):
        """Test that failed call in half_open reopens the circuit."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
            success_threshold=2,
        )

        async def always_fail():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await cb.call(always_fail())

        await asyncio.sleep(0.1)

        await cb.call(asyncio.sleep(0))
        assert cb.state == CircuitState.HALF_OPEN

        with pytest.raises(ValueError):
            await cb.call(always_fail())

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 1

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test that reset returns circuit to initial state."""
        cb = CircuitBreaker(failure_threshold=2)

        async def always_fail():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await cb.call(always_fail())

        with pytest.raises(ValueError):
            await cb.call(always_fail())

        assert cb.state == CircuitState.OPEN

        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.last_failure_time is None


class TestCircuitBreakerTimeout:
    """Tests for circuit breaker with timeout."""

    @pytest.mark.asyncio
    async def test_call_with_timeout_success(self):
        """Test call with timeout that succeeds."""
        cb = CircuitBreaker()

        async def slow_success():
            await asyncio.sleep(0.05)
            return "done"

        result = await cb.call(slow_success(), timeout=1.0)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_call_with_timeout_expires(self):
        """Test call with timeout that expires."""
        cb = CircuitBreaker()

        async def slow_operation():
            await asyncio.sleep(2.0)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await cb.call(slow_operation(), timeout=0.05)

        assert cb.failure_count == 1

    @pytest.mark.asyncio
    async def test_open_circuit_does_not_timeout(self):
        """Test that open circuit rejects immediately without timeout."""
        cb = CircuitBreaker(failure_threshold=1)

        async def fail_once():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await cb.call(fail_once())

        start = time.time()
        with pytest.raises(CircuitBreakerError):
            await cb.call(asyncio.sleep(1.0), timeout=5.0)
        elapsed = time.time() - start

        assert elapsed < 0.5


class TestCircuitBreakerConcurrency:
    """Tests for circuit breaker with concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test circuit breaker with concurrent calls."""
        cb = CircuitBreaker(failure_threshold=10)

        async def task(n):
            await asyncio.sleep(n * 0.01)
            return n * 2

        results = await asyncio.gather(
            cb.call(task(0)),
            cb.call(task(1)),
            cb.call(task(2)),
        )

        assert results == [0, 2, 4]
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_concurrent_failures(self):
        """Test circuit breaker with concurrent failures."""
        cb = CircuitBreaker(failure_threshold=10)

        async def fail_task():
            raise ValueError("test error")

        tasks = [cb.call(fail_task()) for _ in range(5)]

        errors = []
        for task in tasks:
            try:
                await task
            except ValueError:
                errors.append(True)

        assert len(errors) == 5
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 5


class TestCircuitBreakerStateAccessors:
    """Tests for circuit breaker state accessors."""

    def test_state_property(self):
        """Test state property returns correct value."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_failure_count_property(self):
        """Test failure_count property returns correct value."""
        cb = CircuitBreaker()
        assert cb.failure_count == 0

    def test_failure_threshold_property(self):
        """Test failure_threshold property returns correct value."""
        cb = CircuitBreaker(failure_threshold=7)
        assert cb.failure_threshold == 7

    def test_recovery_timeout_property(self):
        """Test recovery_timeout property returns correct value."""
        cb = CircuitBreaker(recovery_timeout=45.0)
        assert cb.recovery_timeout == 45.0

    def test_last_failure_time_none_initially(self):
        """Test last_failure_time is None initially."""
        cb = CircuitBreaker()
        assert cb.last_failure_time is None

    def test_name_property(self):
        """Test name property returns correct value."""
        cb = CircuitBreaker(name="my_cb")
        assert cb.name == "my_cb"

        cb2 = CircuitBreaker()
        assert cb2.name is None
