import asyncio

import pytest

from nodetool.concurrency.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)


class TestAsyncCircuitBreaker:
    """Tests for AsyncCircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Test that circuit breaker starts in CLOSED state."""
        breaker = AsyncCircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        breaker = AsyncCircuitBreaker(
            failure_threshold=10,
            recovery_time=120.0,
            success_threshold=5,
        )
        assert breaker.failure_threshold == 10
        assert breaker.recovery_time == 120.0
        assert breaker.state == CircuitState.CLOSED

    def test_invalid_failure_threshold(self):
        """Test that invalid failure_threshold raises ValueError."""
        with pytest.raises(ValueError, match="failure_threshold must be a positive integer"):
            AsyncCircuitBreaker(failure_threshold=0)

    def test_invalid_recovery_time(self):
        """Test that invalid recovery_time raises ValueError."""
        with pytest.raises(ValueError, match="recovery_time must be a positive number"):
            AsyncCircuitBreaker(recovery_time=0)

    def test_invalid_success_threshold(self):
        """Test that invalid success_threshold raises ValueError."""
        with pytest.raises(ValueError, match="success_threshold must be a positive integer"):
            AsyncCircuitBreaker(success_threshold=0)

    @pytest.mark.asyncio
    async def test_successful_call_stays_closed(self):
        """Test that successful operations keep circuit closed."""
        breaker = AsyncCircuitBreaker(failure_threshold=3)

        async def success_func():
            return "success"

        result = await breaker.call(success_func)

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_success(self):
        """Test context manager with successful operation."""
        breaker = AsyncCircuitBreaker(failure_threshold=3)

        async with breaker:
            result = "success"

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_single_failure_stays_closed(self):
        """Test that single failure doesn't open circuit."""
        breaker = AsyncCircuitBreaker(failure_threshold=3)

        async def fail_func():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await breaker.call(fail_func)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        """Test that circuit opens after reaching failure threshold."""
        breaker = AsyncCircuitBreaker(failure_threshold=3)

        async def fail_func():
            raise ValueError("fail")

        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

    @pytest.mark.asyncio
    async def test_rejects_when_open(self):
        """Test that operations are rejected when circuit is open."""
        breaker = AsyncCircuitBreaker(failure_threshold=2, recovery_time=60.0)

        async def fail_func():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        async def success_func():
            return "success"

        with pytest.raises(CircuitBreakerError):
            await breaker.call(success_func)

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_recovery_time(self):
        """Test that circuit transitions to half-open after recovery time."""
        breaker = AsyncCircuitBreaker(failure_threshold=2, recovery_time=0.01)

        async def fail_func():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        await asyncio.sleep(0.02)

        async def dummy_func():
            return "dummy"

        await breaker.call(dummy_func)

        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_success_threshold_in_half_open(self):
        """Test that circuit closes after success threshold in half-open."""
        breaker = AsyncCircuitBreaker(
            failure_threshold=2,
            recovery_time=0.01,
            success_threshold=2,
        )

        async def fail_func():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        await asyncio.sleep(0.02)

        async def success1():
            return "success1"

        await breaker.call(success1)
        assert breaker.state == CircuitState.HALF_OPEN

        async def success2():
            return "success2"

        await breaker.call(success2)
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reopens_on_failure_in_half_open(self):
        """Test that circuit reopens on failure in half-open state."""
        breaker = AsyncCircuitBreaker(
            failure_threshold=2,
            recovery_time=0.01,
            success_threshold=2,
        )

        async def fail_func():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        await asyncio.sleep(0.02)

        async def dummy():
            return "dummy"

        await breaker.call(dummy)

        assert breaker.state == CircuitState.HALF_OPEN

        with pytest.raises(ValueError):
            await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_filtered_exceptions(self):
        """Test that only monitored exceptions count as failures."""
        breaker = AsyncCircuitBreaker(
            failure_threshold=2,
            monitored_exceptions=(ConnectionError,),
        )

        async def raise_type_error():
            raise TypeError("not monitored")

        with pytest.raises(TypeError):
            await breaker.call(raise_type_error)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_unmonitored_exception_doesnt_count(self):
        """Test that unmonitored exceptions don't increment failure count."""
        breaker = AsyncCircuitBreaker(
            failure_threshold=2,
            monitored_exceptions=(ConnectionError,),
        )

        async def raise_type_error():
            raise TypeError("not monitored")

        for _ in range(3):
            with pytest.raises(TypeError):
                await breaker.call(raise_type_error)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_exception_tracks_failure(self):
        """Test that context manager tracks failures correctly."""
        breaker = AsyncCircuitBreaker(failure_threshold=2)

        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("fail")

        assert breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_context_manager_exception_in_half_open(self):
        """Test context manager behavior in half-open state."""
        breaker = AsyncCircuitBreaker(
            failure_threshold=2,
            recovery_time=0.01,
            success_threshold=2,
        )

        async def fail_func():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        await asyncio.sleep(0.02)

        async def dummy():
            return "dummy"

        await breaker.call(dummy)

        assert breaker.state == CircuitState.HALF_OPEN

        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("fail in half-open")

        assert breaker.state == CircuitState.OPEN

    def test_reset(self):
        """Test that reset returns circuit to closed state."""
        breaker = AsyncCircuitBreaker(failure_threshold=2)

        async def fail_func():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                asyncio.run(breaker.call(fail_func))

        assert breaker.state == CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test that circuit breaker handles concurrent access correctly."""
        breaker = AsyncCircuitBreaker(failure_threshold=10)

        async def fail_operation():
            async def fail_func():
                raise ValueError("fail")

            await breaker.call(fail_func)

        async def success_operation():
            async def success_func():
                return "success"

            return await breaker.call(success_func)

        tasks = [fail_operation() for _ in range(5)] + [success_operation()]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        failures = sum(1 for r in results if isinstance(r, ValueError))
        success_count = sum(1 for r in results if r == "success")

        assert failures == 5
        assert success_count == 1
        assert len(results) == 6


class TestCircuitBreakerError:
    """Tests for CircuitBreakerError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = CircuitBreakerError()
        assert str(error) == "Circuit breaker is open"

    def test_custom_message(self):
        """Test custom error message."""
        error = CircuitBreakerError("Custom message")
        assert str(error) == "Custom message"


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_states_exist(self):
        """Test that all expected states exist."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"
