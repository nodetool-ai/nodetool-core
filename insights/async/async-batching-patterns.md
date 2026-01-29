# Async Batching Utilities

**Insight**: Added `batched_async_iterable` and `process_in_batches` utilities for efficient async batch processing.

**Rationale**: Processing large datasets in batches improves memory efficiency and allows controlled parallelism. The new utilities complement existing concurrency tools (`AsyncSemaphore`, `gather_with_concurrency`) by providing batch-specific functionality.

**Example**:
```python
from nodetool.concurrency import batched_async_iterable, process_in_batches

# Async iteration over batches
async for batch in batched_async_iterable(items, batch_size=100):
    await process_batch(batch)

# Process with concurrent batch execution
results = await process_in_batches(
    items=urls,
    processor=fetch_batch,
    batch_size=10,
    max_concurrent=3,
)
```

**Impact**: Useful for batch API calls, database operations, and file processing with memory-efficient chunking.

**Files**:
- `src/nodetool/concurrency/batching.py`
- `tests/concurrency/test_batching.py`

**Date**: 2026-01-15
