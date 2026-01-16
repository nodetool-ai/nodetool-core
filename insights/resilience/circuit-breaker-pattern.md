# Circuit Breaker Pattern for Resilience

**Insight**: Added `CircuitBreaker` class to the concurrency module to prevent cascading failures in distributed AI systems.

**Rationale**: When AI workflows make calls to external services (APIs, databases, model providers), failures can cascade if services are overloaded. The circuit breaker pattern provides:
- Automatic detection of service failures
- Fast failure responses when circuit is open
- Automatic recovery attempts after timeout
- Protection against thundering herd on recovering services

**Example**:
```python
from nodetool.concurrency import CircuitBreaker, CircuitBreakerError

cb = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
    success_threshold=3
)

async with cb:
    result = await call_external_api()
```

**Impact**: Combined with existing `RetryPolicy` and `AsyncRateLimiter`, provides comprehensive resilience patterns for AI workflow operations.

**Files**: `src/nodetool/concurrency/circuit_breaker.py`, `tests/concurrency/test_circuit_breaker.py`

**Date**: 2026-01-16
