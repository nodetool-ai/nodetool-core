# Secret Cache Thread Safety Issue

**Problem**: The `_SECRET_CACHE` dictionary in `secret_helper.py` was accessed from multiple async functions without any locking mechanism, creating race conditions in concurrent access scenarios.

**Solution**: Added an `asyncio.Lock` (`_SECRET_CACHE_LOCK`) and wrapped all cache read/write operations with `async with _SECRET_CACHE_LOCK:` to ensure thread-safe access in async contexts.

**Why**: The secret cache is a shared mutable state accessed by multiple concurrent async tasks. Without proper synchronization, race conditions could cause:
- Lost updates (two coroutines reading and writing simultaneously)
- Inconsistent reads (reading partially updated state)
- KeyError exceptions during iteration

**Files**:
- `src/nodetool/security/secret_helper.py`

**Date**: 2026-02-16
