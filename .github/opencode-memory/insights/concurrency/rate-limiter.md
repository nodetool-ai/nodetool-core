# AsyncRateLimiter Implementation

**Insight**: Implemented a token bucket rate limiter for the concurrency module to complement existing async utilities.

**Rationale**: External API calls often have rate limits that need to be respected. The `AsyncRateLimiter` provides a simple, async-first rate limiting solution that works well with the existing `AsyncSemaphore`, retry, and timeout utilities.

**Example**:
```python
from nodetool.concurrency import AsyncRateLimiter

# Allow 10 requests per second
limiter = AsyncRateLimiter(rate=10, interval=1.0)

# Use as context manager
async with limiter:
    await make_api_call()

# Or use acquire directly
result = await limiter.acquire()
if result.allowed:
    await make_api_call()
```

**Impact**: Provides a foundational building block for rate-limited API calls to external services.

**Files**:
- `src/nodetool/concurrency/rate_limiter.py`
- `tests/concurrency/test_rate_limiter.py`

**Date**: 2026-01-14
