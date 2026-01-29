# AsyncBuffer for Streaming Data

**Insight**: Adding `AsyncBuffer` to the concurrency utilities provides a bounded, thread-safe buffer for async data streams with built-in backpressure handling.

**Rationale**: Streaming data scenarios require controlled buffering to prevent memory exhaustion while maintaining throughput. The `AsyncBuffer` class complements existing utilities like `batched_async_iterable` by providing explicit control over memory usage and producer-consumer coordination.

**Example**:

```python
from nodetool.concurrency import AsyncBuffer

buffer = AsyncBuffer(max_size=100, block_on_full=True, timeout=30.0)

async def producer():
    for data in streaming_source():
        await buffer.put(data)

async def consumer():
    async for item in buffer:
        await process(item)

await asyncio.gather(producer(), consumer())
```

**Impact**:
- Prevents memory exhaustion in high-volume streaming scenarios
- Provides backpressure signaling via blocking/exception behavior
- Supports graceful shutdown with `aclose()` and `flush()`
- Enables efficient batch processing with `get_batch()` method

**Files**:
- `src/nodetool/concurrency/async_buffer.py`
- `src/nodetool/concurrency/__init__.py`
- `tests/concurrency/test_async_buffer.py`

**Date**: 2026-01-20
