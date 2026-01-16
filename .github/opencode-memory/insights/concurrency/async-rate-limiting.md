# Async Rate Limiting with Token Bucket

**Insight**: Implemented `AsyncTokenBucket` and `AsyncRateLimiter` in `src/nodetool/concurrency/rate_limit.py` using the token bucket algorithm for async rate limiting.

**Rationale**: Rate limiting is essential for controlling API usage, preventing abuse, and respecting rate limits imposed by external services. The token bucket algorithm allows controlled burstiness while maintaining a long-term rate limit - ideal for AI API interactions.

**Example**:
```python
from nodetool.concurrency import AsyncRateLimiter

# 10 requests per second with burst capacity of 20
limiter = AsyncRateLimiter(rate=10, capacity=20)

async with limiter:
    await make_api_request()  # Automatically rate-limited
```

**Impact**: Complements `AsyncSemaphore` (which limits concurrent tasks) with the ability to limit operations over time. Essential for AI API rate limiting scenarios.

**Files**:
- `src/nodetool/concurrency/rate_limit.py` - Implementation
- `src/nodetool/concurrency/__init__.py` - Exports
- `tests/concurrency/test_rate_limit.py` - Tests

**Date**: 2026-01-16
