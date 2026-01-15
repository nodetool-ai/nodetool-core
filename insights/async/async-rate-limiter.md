# AsyncRateLimiter Token Bucket Implementation

**Insight**: Implemented an async rate limiter using the token bucket algorithm for time-based rate limiting.

**Rationale**: The token bucket algorithm is well-suited for rate limiting because it:
- Allows burst traffic up to a configurable limit
- Smoothly refills tokens over time
- Works well for API rate limiting scenarios

**Implementation Details**:
- `AsyncRateLimiter` class with configurable `max_rate`, `time_period`, and `burst`
- Token refill happens lazily on each `acquire()` call
- Thread-safe using `asyncio.Lock`
- Supports both context manager and decorator usage patterns

**Files**:
- `src/nodetool/concurrency/rate_limiter.py`
- `src/nodetool/concurrency/__init__.py`
- `tests/concurrency/test_rate_limiter.py`

**Date**: 2026-01-15
