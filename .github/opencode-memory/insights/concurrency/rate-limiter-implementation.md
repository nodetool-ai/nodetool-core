# RateLimiter Implementation

**Insight**: Implemented a token bucket rate limiter with async support for API clients and resource management.

**Rationale**: Rate limiting is essential for controlling the rate of operations, especially when interacting with external APIs that have rate limits. The token bucket algorithm allows for burst traffic while maintaining a steady average rate.

**Implementation Details**:
- Token bucket algorithm with configurable rate (tokens/second) and burst (bucket capacity)
- `acquire()`: Non-blocking, returns True if tokens available, False otherwise
- `acquire_or_wait()`: Blocking, waits until tokens are available
- Context manager support for convenient usage
- `rate_limited_gather()`: Helper for rate-limited concurrent execution
- Thread-safe using asyncio.Lock

**Files**:
- `src/nodetool/concurrency/rate_limiter.py` - Implementation
- `tests/concurrency/test_rate_limiter.py` - Comprehensive tests

**Example Usage**:
```python
limiter = RateLimiter(rate=10, burst=5)

# Using context manager
async with limiter:
    await make_api_call()

# Using acquire with wait
await limiter.acquire_or_wait()
await make_api_call()

# Rate-limited concurrent execution
results = await rate_limited_gather(
    [fetch_url(url) for url in urls],
    rate=10,  # 10 requests per second
    burst=5,  # allow bursts up to 5
)
```

**Date**: 2026-01-16
