import asyncio

import pytest

from nodetool.concurrency.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    with_circuit_breaker,
)


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_initial_state_is_closed(self):
        """Test that circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=3)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self):
        """Test that circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        async def fail():
            raise ValueError("fail")

        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    await fail()

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_rejects_when_open(self):
        """Test that requests are rejected when circuit is open."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            async with breaker:
                await fail()

        with pytest.raises(CircuitBreakerError):
            await breaker.acquire()

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        """Test that circuit transitions to HALF_OPEN after recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            async with breaker:
                await fail()

        assert breaker.state == CircuitState.OPEN

        await asyncio.sleep(0.02)

        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_success_in_half_open(self):
        """Test that circuit closes after successes in HALF_OPEN."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=2,
        )

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            async with breaker:
                await fail()

        await asyncio.sleep(0.02)

        assert breaker.state == CircuitState.HALF_OPEN

        await breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN

        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_trips_back_to_open_on_failure_in_half_open(self):
        """Test that circuit trips back to OPEN on failure in HALF_OPEN."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=2,
        )

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            async with breaker:
                await fail()

        await asyncio.sleep(0.02)

        assert breaker.state == CircuitState.HALF_OPEN

        with pytest.raises(ValueError):
            async with breaker:
                await fail()

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerRecordOperations:
    """Tests for failure/success recording."""

    @pytest.mark.asyncio
    async def test_record_success_resets_failure_count(self):
        """Test that success resets failure count in CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=3)

        await breaker.record_failure(ValueError("fail"))
        await breaker.record_failure(ValueError("fail"))

        assert breaker.failure_count == 2

        await breaker.record_success()

        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_record_failure_increments_counter(self):
        """Test that failures increment the counter."""
        breaker = CircuitBreaker(failure_threshold=3)

        await breaker.record_failure(ValueError("fail"))
        assert breaker.failure_count == 1

        await breaker.record_failure(ValueError("fail"))
        assert breaker.failure_count == 2

    @pytest.mark.asyncio
    async def test_non_failure_exceptions_not_counted(self):
        """Test that non-specified exceptions are not counted."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            exception_type=ConnectionError,
        )

        with pytest.raises(TypeError):
            async with breaker:
                raise TypeError("not a connection error")

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0


class TestCircuitBreakerDecorator:
    """Tests for circuit breaker decorator usage."""

    @pytest.mark.asyncio
    async def test_decorator_closes_on_success(self):
        """Test that decorator closes after successful execution."""
        call_count = 0

        @CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await succeed()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_opens_on_failure(self):
        """Test that decorator opens circuit on repeated failures."""
        call_count = 0
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)

        @breaker
        async def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("fail")
            return "success"

        with pytest.raises(ValueError):
            await fail_twice()

        with pytest.raises(ValueError):
            await fail_twice()

        assert call_count == 2

        with pytest.raises(CircuitBreakerError):
            await fail_twice()

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @CircuitBreaker(failure_threshold=1)
        async def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestWithCircuitBreaker:
    """Tests for the with_circuit_breaker function."""

    @pytest.mark.asyncio
    async def test_success_passes_through(self):
        """Test that successful calls pass through."""
        breaker = CircuitBreaker(failure_threshold=3)

        result = await with_circuit_breaker(
            lambda: asyncio.sleep(0, result="success"),
            breaker,
        )

        assert result == "success"

    @pytest.mark.asyncio
    async def test_failure_is_tracked(self):
        """Test that failures are tracked correctly."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await with_circuit_breaker(fail, breaker)

        assert breaker.state == CircuitState.CLOSED

        with pytest.raises(ValueError):
            await with_circuit_breaker(fail, breaker)

        assert breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError):
            await with_circuit_breaker(lambda: asyncio.sleep(0), breaker)

    @pytest.mark.asyncio
    async def test_circuit_open_raises_error(self):
        """Test that CircuitBreakerError is raised when open."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await with_circuit_breaker(fail, breaker)

        assert breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError):
            await with_circuit_breaker(lambda: asyncio.sleep(0), breaker)


class TestCircuitBreakerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_failure_threshold(self):
        """Test that invalid failure_threshold raises ValueError."""
        with pytest.raises(ValueError, match="failure_threshold must be"):
            CircuitBreaker(failure_threshold=0)

        with pytest.raises(ValueError, match="failure_threshold must be"):
            CircuitBreaker(failure_threshold=-1)

    def test_invalid_recovery_timeout(self):
        """Test that invalid recovery_timeout raises ValueError."""
        with pytest.raises(ValueError, match="recovery_timeout must be"):
            CircuitBreaker(recovery_timeout=0)

        with pytest.raises(ValueError, match="recovery_timeout must be"):
            CircuitBreaker(recovery_timeout=-1)

    @pytest.mark.asyncio
    async def test_context_manager_exception_releases(self):
        """Test that context manager releases properly on exception."""
        breaker = CircuitBreaker(failure_threshold=3)

        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("test error")

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_allows_one_request(self):
        """Test that HALF_OPEN state allows exactly one request through."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=2,
        )

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await with_circuit_breaker(fail, breaker)

        assert breaker.state == CircuitState.OPEN

        await asyncio.sleep(0.02)

        assert breaker.state == CircuitState.HALF_OPEN

        result = await with_circuit_breaker(
            lambda: asyncio.sleep(0, result="test"),
            breaker,
        )
        assert result == "test"

        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_failure_count_resets_on_success(self):
        """Test that failure count resets after success."""
        breaker = CircuitBreaker(failure_threshold=3)

        await breaker.record_failure(ValueError("fail"))
        await breaker.record_failure(ValueError("fail"))
        assert breaker.failure_count == 2

        await breaker.record_success()
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_tripped_during_half_open_logs_error(self):
        """Test that tripping during HALF_OPEN is logged."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=2,
        )

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await with_circuit_breaker(fail, breaker)

        assert breaker.state == CircuitState.OPEN

        await asyncio.sleep(0.02)

        assert breaker.state == CircuitState.HALF_OPEN

        with pytest.raises(ValueError):
            await with_circuit_breaker(fail, breaker)

        assert breaker.state == CircuitState.OPEN
