# Circuit Breaker Pattern for Resilience

**Insight**: Added Circuit Breaker pattern implementation to the concurrency module for protecting against cascading failures in distributed systems.

**Rationale**: The circuit breaker complements existing retry and timeout utilities by preventing an application from repeatedly trying to execute an operation that's likely to fail. This gives downstream services time to recover and prevents resource exhaustion.

**Usage**:
```python
from nodetool.concurrency import CircuitBreaker

cb = CircuitBreaker(
    failure_threshold=5,  # Open after 5 failures
    recovery_timeout=30.0,  # Try recovery after 30s
    success_threshold=3,  # Need 3 successes to close
)

# Use as context manager
async with cb:
    await make_external_api_call()

# Use as decorator
@CircuitBreaker(failure_threshold=3)
async def fragile_service():
    ...

# Use execute method directly
result = await cb.execute(lambda: some_operation())
```

**States**:
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Failure threshold reached, requests are rejected immediately with `CircuitBreakerError`
- **HALF_OPEN**: Testing if service recovered, limited requests allowed

**Files**:
- `src/nodetool/concurrency/circuit_breaker.py`
- `tests/concurrency/test_circuit_breaker.py`

**Date**: 2026-01-15
