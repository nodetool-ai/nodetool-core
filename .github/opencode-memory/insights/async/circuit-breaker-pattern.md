# Circuit Breaker Pattern for Async Python

**Insight**: The circuit breaker pattern prevents cascade failures in distributed systems by tracking failures and temporarily blocking requests to failing services.

**Rationale**: When a downstream service fails, repeatedly sending requests can:
- Overload the failing service
- Cause timeouts and resource exhaustion
- Create cascade failures across the system

The circuit breaker pattern addresses this by:
1. Tracking failure counts
2. Opening the circuit after N failures (rejecting all requests)
3. Testing recovery with limited requests (HALF_OPEN)
4. Closing the circuit when the service recovers

**Example**:
```python
from nodetool.concurrency import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
    success_threshold=3,
)

async def fetch_user_data(user_id: str) -> dict:
    async with breaker:
        return await api.get(f"/users/{user_id}")

# Or use as a decorator
@breaker
async def fetch_order_data(order_id: str) -> dict:
    return await api.get(f"/orders/{order_id}")
```

**Impact**:
- Prevents cascading failures
- Reduces load on failing services
- Enables faster recovery detection
- Complements retry policies for transient failures

**Files**: `src/nodetool/concurrency/circuit_breaker.py`, `tests/concurrency/test_circuit_breaker.py`

**Date**: 2026-01-14
