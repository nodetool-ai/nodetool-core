# Circuit Breaker Pattern for Fault Tolerance

**Insight**: Implemented the circuit breaker pattern as a core concurrency utility to prevent cascading failures in distributed systems.

**Rationale**: The circuit breaker pattern is essential for building resilient systems that interact with external services. It:
- Prevents cascading failures by failing fast when a service is unavailable
- Allows systems to recover automatically after transient failures
- Provides better observability into service health
- Complements retry utilities for comprehensive fault tolerance

**Implementation Details**:
- `AsyncCircuitBreaker`: Main class implementing the pattern with three states (CLOSED, OPEN, HALF_OPEN)
- `CircuitState` enum: Defines the three possible states
- `CircuitBreakerError`: Exception raised when operations are rejected due to open circuit

**Key Features**:
- Configurable failure threshold before opening
- Automatic recovery time before attempting half-open state
- Success threshold in half-open state to fully close
- Support for filtered exception monitoring
- Thread-safe async implementation using asyncio.Lock
- Context manager support for clean usage

**Example Usage**:
```python
breaker = AsyncCircuitBreaker(
    failure_threshold=5,
    recovery_time=60.0,
    success_threshold=3,
)

async with breaker:
    return await make_api_request()
```

**Files**: `src/nodetool/concurrency/circuit_breaker.py`

**Date**: 2026-01-16
