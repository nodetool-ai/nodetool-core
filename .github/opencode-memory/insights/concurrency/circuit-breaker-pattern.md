# Circuit Breaker Pattern for System Resilience

**Insight**: The circuit breaker pattern prevents cascading failures by stopping requests to failing services and allowing them to recover.

**Rationale**: When external services become unavailable, repeated failed requests can:
- Waste resources on hopeless retries
- Cause cascading failures across the system
- Prevent the downstream service from recovering

**Example**:
```python
from nodetool.concurrency import CircuitBreaker, CircuitBreakerConfig

# Create a circuit breaker that opens after 5 failures
breaker = CircuitBreaker(
    CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=30.0,
        success_threshold=3,
    )
)

# Use with retry for robust external service calls
async with breaker:
    result = await retry_with_exponential_backoff(
        lambda: call_external_api(),
        retryable_exceptions=(ServiceUnavailableError,),
    )
```

**Impact**:
- Prevents resource exhaustion from repeated failures
- Allows downstream services time to recover
- Provides fast failure instead of slow timeout delays

**Files**: `src/nodetool/concurrency/circuit_breaker.py`, `tests/concurrency/test_circuit_breaker.py`

**Date**: 2026-01-17
