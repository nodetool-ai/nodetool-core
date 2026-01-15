import asyncio

import pytest

from nodetool.concurrency.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)


class TestCircuitBreakerInitialization:
    """Tests for CircuitBreaker initialization."""

    def test_default_initialization(self):
        """Test default CircuitBreaker values."""
        cb = CircuitBreaker()
        assert cb._failure_threshold == 5
        assert cb._recovery_timeout == 30.0
        assert cb._success_threshold == 3
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_custom_initialization(self):
        """Test CircuitBreaker with custom values."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10.0,
            success_threshold=2,
        )
        assert cb._failure_threshold == 3
        assert cb._recovery_timeout == 10.0
        assert cb._success_threshold == 2

    def test_invalid_failure_threshold(self):
        """Test that invalid failure_threshold raises ValueError."""
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreaker(failure_threshold=0)

        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreaker(failure_threshold=-1)

    def test_invalid_recovery_timeout(self):
        """Test that invalid recovery_timeout raises ValueError."""
        with pytest.raises(ValueError, match="recovery_timeout must be positive"):
            CircuitBreaker(recovery_timeout=0)

    def test_invalid_success_threshold(self):
        """Test that invalid success_threshold raises ValueError."""
        with pytest.raises(ValueError, match="success_threshold must be positive"):
            CircuitBreaker(success_threshold=0)


class TestCircuitBreakerStates:
    """Tests for CircuitBreaker state transitions."""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self):
        """Test that circuit starts in closed state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        assert not cb.is_open
        assert not cb.is_half_open

    @pytest.mark.asyncio
    async def test_state_properties(self):
        """Test state property accessors."""
        cb = CircuitBreaker(failure_threshold=1)

        assert cb.is_closed
        assert not cb.is_open
        assert not cb.is_half_open

    @pytest.mark.asyncio
    async def test_opens_on_failure_threshold(self):
        """Test that circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.state == CircuitState.OPEN
        assert cb.is_open
        assert not cb.is_closed

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        """Test that circuit transitions to half-open after recovery timeout."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
        )

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.is_open

        await asyncio.sleep(0.02)

        await cb.execute(lambda: None)
        assert cb.is_half_open


class TestCircuitBreakerExecution:
    """Tests for CircuitBreaker execution methods."""

    @pytest.mark.asyncio
    async def test_successful_execution_closed_state(self):
        """Test that successful execution works in closed state."""
        cb = CircuitBreaker()

        async def success_func():
            return "success"

        result = await cb.execute(success_func)
        assert result == "success"
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_failure_counts_toward_threshold(self):
        """Test that failures are counted toward the threshold."""
        cb = CircuitBreaker(failure_threshold=2)

        for _ in range(2):
            with pytest.raises(ValueError, match="fail"):
                await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.failure_count == 2
        assert cb.is_open

    @pytest.mark.asyncio
    async def test_rejects_when_open(self):
        """Test that execution is rejected when circuit is open."""
        cb = CircuitBreaker(failure_threshold=1)

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.is_open

        with pytest.raises(CircuitBreakerError):
            await cb.execute(lambda: "success")

    @pytest.mark.asyncio
    async def test_success_in_half_open_closes_circuit(self):
        """Test that success in half-open state closes the circuit."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=1,
        )

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        await asyncio.sleep(0.02)

        result = await cb.execute(lambda: "success")
        assert result == "success"
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens_circuit(self):
        """Test that failure in half-open state reopens the circuit."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=2,
        )

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        await asyncio.sleep(0.02)

        result = await cb.execute(lambda: "first success")
        assert result == "first success"
        assert cb.is_half_open

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.is_open

    @pytest.mark.asyncio
    async def test_success_threshold_closes_circuit(self):
        """Test that reaching success threshold closes the circuit."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=2,
        )

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        await asyncio.sleep(0.02)

        await cb.execute(lambda: None)
        assert cb.is_half_open

        result = await cb.execute(lambda: "success")
        assert result == "success"
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """Test that reset clears all state."""
        cb = CircuitBreaker(failure_threshold=1)

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.is_open
        assert cb.failure_count == 1

        cb.reset()

        assert cb.is_closed
        assert cb.failure_count == 0


class TestCircuitBreakerContextManager:
    """Tests for CircuitBreaker as async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_success(self):
        """Test context manager with successful operation."""
        cb = CircuitBreaker()

        result = None
        async with cb:
            result = "success"

        assert result == "success"
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_context_manager_failure(self):
        """Test context manager with failed operation."""
        cb = CircuitBreaker(failure_threshold=2)

        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("fail")

        assert cb.failure_count == 1
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_context_manager_opens_circuit(self):
        """Test that repeated failures in context manager open the circuit."""
        cb = CircuitBreaker(failure_threshold=2)

        for _ in range(2):
            with pytest.raises(ValueError):
                async with cb:
                    raise ValueError("fail")

        assert cb.is_open

    @pytest.mark.asyncio
    async def test_context_manager_rejects_when_open(self):
        """Test that context manager rejects when circuit is open."""
        cb = CircuitBreaker(failure_threshold=1)

        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("fail")

        with pytest.raises(CircuitBreakerError):
            async with cb:
                pass


class TestCircuitBreakerDecorator:
    """Tests for CircuitBreaker as a decorator."""

    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test decorator with successful function."""
        cb = CircuitBreaker(failure_threshold=3)

        @cb
        async def successful_func():
            return "result"

        result = await successful_func()
        assert result == "result"
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_decorator_opens_circuit(self):
        """Test decorator opens circuit after failures."""
        cb = CircuitBreaker(failure_threshold=2)

        @cb
        async def failing_func():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await failing_func()

        assert cb.is_open

        with pytest.raises(CircuitBreakerError):
            await failing_func()

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @CircuitBreaker(failure_threshold=1)
        async def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestCircuitBreakerExpectedException:
    """Tests for CircuitBreaker with custom expected exception."""

    @pytest.mark.asyncio
    async def test_custom_expected_exception(self):
        """Test CircuitBreaker with custom expected exception type."""
        cb = CircuitBreaker(
            failure_threshold=2,
            expected_exception=ConnectionError,
        )

        for _ in range(2):
            with pytest.raises(ConnectionError):
                await cb.execute(lambda: (_ for _ in ()).throw(ConnectionError("fail")))

        assert cb.is_open

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        """Test that circuit transitions to half-open after recovery timeout."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
        )

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.is_open

        await asyncio.sleep(0.02)

        await cb.execute(lambda: None)
        assert cb.is_half_open

    @pytest.mark.asyncio
    async def test_success_in_half_open_closes_circuit(self):
        """Test that success in half-open state closes the circuit."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=1,
        )

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        await asyncio.sleep(0.02)

        result = await cb.execute(lambda: "success")
        assert result == "success"
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens_circuit(self):
        """Test that failure in half-open state reopens the circuit."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=2,
        )

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        await asyncio.sleep(0.02)

        result = await cb.execute(lambda: "first success")
        assert result == "first success"
        assert cb.is_half_open

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert cb.is_open

    @pytest.mark.asyncio
    async def test_success_threshold_closes_circuit(self):
        """Test that reaching success threshold closes the circuit."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=2,
        )

        with pytest.raises(ValueError):
            await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))

        await asyncio.sleep(0.02)

        await cb.execute(lambda: None)
        assert cb.is_half_open

        result = await cb.execute(lambda: "success")
        assert result == "success"
        assert cb.is_closed


class TestCircuitBreakerConcurrency:
    """Tests for CircuitBreaker thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test concurrent execution through circuit breaker."""
        cb = CircuitBreaker(failure_threshold=10)

        async def task(n: int) -> int:
            async def compute() -> int:
                return n * 2

            return await cb.execute(compute)

        results = await asyncio.gather(*(task(i) for i in range(5)))
        assert results == [0, 2, 4, 6, 8]
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_concurrent_failures(self):
        """Test concurrent failures are properly counted."""
        cb = CircuitBreaker(failure_threshold=3)

        async def fail_once(n):
            if n == 0:
                return await cb.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))
            return n

        with pytest.raises(ValueError):
            await asyncio.gather(*(fail_once(i) for i in range(3)))

        assert cb.failure_count >= 1
