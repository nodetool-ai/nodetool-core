import asyncio
import time

import pytest

from nodetool.concurrency.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    MultiCircuitBreaker,
)


class TestCircuitBreakerStates:
    """Tests for circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_starts_in_closed_state(self):
        """Test that circuit breaker starts in closed state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_successful_calls_keep_circuit_closed(self):
        """Test that successful calls maintain closed state."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))

        async def success():
            return "ok"

        result = await breaker.call(success)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self):
        """Test that circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))

        async def fail():
            raise ValueError("fail")

        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_rejects_calls_when_open(self):
        """Test that calls are rejected when circuit is open."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))

        async def fail():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail)

        with pytest.raises(CircuitBreakerError) as exc_info:
            await breaker.call(lambda: asyncio.sleep(0))

        assert exc_info.value.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        """Test that circuit transitions to half-open after timeout."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.05))

        async def fail():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail)

        assert breaker.state == CircuitState.OPEN

        await asyncio.sleep(0.1)

        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_successful_calls_close_circuit_from_half_open(self):
        """Test that successful calls in half-open close the circuit."""
        breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=2,
                success_threshold=2,
                timeout_seconds=0.01,
                half_open_max_calls=5,
            )
        )

        async def fail():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail)

        await asyncio.sleep(0.05)

        async def success():
            return "ok"

        for _ in range(2):
            result = await breaker.call(success)
            assert result == "ok"

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_single_failure_reopens_from_half_open(self):
        """Test that a single failure in half-open reopens the circuit."""
        breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=2,
                success_threshold=3,
                timeout_seconds=0.01,
                half_open_max_calls=5,
            )
        )

        async def fail():
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail)

        await asyncio.sleep(0.05)

        async def success():
            return "ok"

        result = await breaker.call(success)
        assert result == "ok"

        with pytest.raises(ValueError):
            await breaker.call(fail)

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerStatistics:
    """Tests for circuit breaker statistics tracking."""

    @pytest.mark.asyncio
    async def test_tracks_total_calls(self):
        """Test that total calls are tracked."""
        breaker = CircuitBreaker()

        async def success():
            return "ok"

        await breaker.call(success)
        await breaker.call(success)

        assert breaker.stats.total_calls == 2

    @pytest.mark.asyncio
    async def test_tracks_successful_calls(self):
        """Test that successful calls are tracked."""
        breaker = CircuitBreaker()

        async def success():
            return "ok"

        await breaker.call(success)

        assert breaker.stats.successful_calls == 1

    @pytest.mark.asyncio
    async def test_tracks_failed_calls(self):
        """Test that failed calls are tracked."""
        breaker = CircuitBreaker()

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await breaker.call(fail)

        assert breaker.stats.failed_calls == 1

    @pytest.mark.asyncio
    async def test_tracks_rejected_calls(self):
        """Test that rejected calls are tracked."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await breaker.call(fail)

        with pytest.raises(CircuitBreakerError):
            await breaker.call(lambda: asyncio.sleep(0))

        assert breaker.stats.rejected_calls == 1

    @pytest.mark.asyncio
    async def test_tracks_last_failure(self):
        """Test that last failure is tracked."""
        breaker = CircuitBreaker()

        async def fail():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await breaker.call(fail)

        assert breaker.stats.last_failure_exception is not None
        assert "test error" in str(breaker.stats.last_failure_exception)


class TestCircuitBreakerExceptionFilter:
    """Tests for exception filtering."""

    @pytest.mark.asyncio
    async def test_filtered_exceptions_not_counted(self):
        """Test that filtered exceptions don't count toward failure threshold."""
        breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=3,
                exception_filter=lambda e: not isinstance(e, ValueError),
            )
        )

        async def raise_value_error():
            raise ValueError("ignored")

        for _ in range(5):
            with pytest.raises(ValueError):
                await breaker.call(raise_value_error)

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_unfiltered_exceptions_counted(self):
        """Test that unfiltered exceptions count toward failure threshold."""
        breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=3,
                exception_filter=lambda e: isinstance(e, ValueError),
            )
        )

        async def raise_value_error():
            raise ValueError("counted")

        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(raise_value_error)

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerContextManager:
    """Tests for context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager_success(self):
        """Test context manager with successful operation."""
        async with CircuitBreaker() as breaker:
            result = await breaker.call(lambda: asyncio.sleep(0))
            assert result is None
            assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_context_manager_failure_tracked(self):
        """Test that failures in context manager are tracked."""
        async with CircuitBreaker(CircuitBreakerConfig(failure_threshold=2)) as breaker:

            async def fail():
                raise ValueError("fail")

            with pytest.raises(ValueError):
                await breaker.call(fail)

        assert breaker.stats.failed_calls == 1

    @pytest.mark.asyncio
    async def test_context_manager_exception_in_body(self):
        """Test that exceptions in with body are tracked by __aexit__."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))

        try:
            async with breaker:
                raise ValueError("direct exception")
        except ValueError:
            pass  # Exception should propagate

        assert breaker.stats.failed_calls == 1

    @pytest.mark.asyncio
    async def test_context_manager_direct_failure_opens_circuit(self):
        """Test that direct exceptions in with body open the circuit after threshold."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))

        try:
            async with breaker:
                raise ValueError("direct exception")
        except ValueError:
            pass

        assert breaker.stats.failed_calls == 1
        assert breaker.state == CircuitState.CLOSED  # Only 1 failure, threshold is 2

        try:
            async with breaker:
                raise ValueError("second exception")
        except ValueError:
            pass

        assert breaker.stats.failed_calls == 2
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_seconds == 60.0
        assert config.half_open_max_calls == 3
        assert config.exception_filter is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            timeout_seconds=120.0,
            half_open_max_calls=10,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.timeout_seconds == 120.0
        assert config.half_open_max_calls == 10


class TestCircuitBreakerEdgeCases:
    """Tests for edge cases in circuit breaker."""

    @pytest.mark.asyncio
    async def test_zero_failure_threshold(self):
        """Test circuit breaker with zero failure threshold."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=0))

        async def success():
            return "ok"

        with pytest.raises(CircuitBreakerError) as exc_info:
            await breaker.call(success)

        assert exc_info.value.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_limit(self):
        """Test that half-open call limit is enforced."""
        breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                success_threshold=10,
                timeout_seconds=0.01,
                half_open_max_calls=2,
            )
        )

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await breaker.call(fail)

        await asyncio.sleep(0.05)

        async def success():
            return "ok"

        await breaker.call(success)
        await breaker.call(success)

        with pytest.raises(CircuitBreakerError):
            await breaker.call(success)

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test circuit breaker with concurrent calls."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, half_open_max_calls=10))

        async def fail():
            raise ValueError("fail")

        tasks = [breaker.call(fail) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        rejected_count = sum(1 for r in results if isinstance(r, CircuitBreakerError))
        failed_count = sum(1 for r in results if isinstance(r, ValueError))

        assert rejected_count + failed_count == 5

    @pytest.mark.asyncio
    async def test_repr(self):
        """Test string representation."""
        breaker = CircuitBreaker()
        repr_str = repr(breaker)
        assert "CircuitBreaker" in repr_str
        assert "closed" in repr_str


class TestMultiCircuitBreaker:
    """Tests for MultiCircuitBreaker."""

    @pytest.mark.asyncio
    async def test_separate_circuits_for_different_keys(self):
        """Test that different keys have separate circuits."""
        multi = MultiCircuitBreaker(CircuitBreakerConfig(failure_threshold=2, timeout_seconds=60.0))

        async def fail_service_a():
            raise ValueError("service A down")

        async def success_service_b():
            return "service B OK"

        for _ in range(2):
            with pytest.raises(ValueError):
                await multi.call("service_a", fail_service_a)

        assert multi.get_stats("service_a") is not None
        assert multi.get_stats("service_b") is None

        result = await multi.call("service_b", success_service_b)
        assert result == "service B OK"

    @pytest.mark.asyncio
    async def test_get_all_stats(self):
        """Test getting stats for all circuits."""
        multi = MultiCircuitBreaker()

        async def success():
            return "ok"

        await multi.call("key1", success)
        await multi.call("key2", success)
        await multi.call("key1", success)

        stats = multi.get_all_stats()
        assert "key1" in stats
        assert "key2" in stats
        assert stats["key1"].total_calls == 2
        assert stats["key2"].total_calls == 1

    @pytest.mark.asyncio
    async def test_reset_specific_key(self):
        """Test resetting a specific circuit."""
        multi = MultiCircuitBreaker(CircuitBreakerConfig(failure_threshold=1, timeout_seconds=60.0))

        async def fail():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await multi.call("key1", fail)

        multi.reset("key1")

        assert multi.get_stats("key1") is None

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """Test resetting all circuits."""
        multi = MultiCircuitBreaker()

        async def success():
            return "ok"

        await multi.call("key1", success)
        await multi.call("key2", success)

        multi.reset()

        assert multi.get_all_stats() == {}
        assert len(multi._breakers) == 0

    @pytest.mark.asyncio
    async def test_repr(self):
        """Test string representation."""
        multi = MultiCircuitBreaker()
        repr_str = repr(multi)
        assert "MultiCircuitBreaker" in repr_str
