# Async Chunks Utility

**Insight**: `async_chunks` splits async iterables into fixed-size chunks, complementing existing async iterator utilities.

**Rationale**: While `batched_async_iterable` in batching.py is for processing (applies functions to batches), `async_chunks` is a pure transformation that yields chunks as lists. This separation of concerns makes the API clearer - use chunks when you just need the data grouped, and batched processing when you need to apply operations.

**Example**:
```python
from nodetool.concurrency import async_chunks, async_list

# Split a stream of records into batches for database insertion
async def process_records(record_stream):
    batches = async_chunks(record_stream, n=100)

    async for batch in batches:
        await db.bulk_insert(batch)

# Or collect all chunks at once
chunks = await async_list(async_chunks(generate_items(), chunk_size=50))
# Returns: [[item0, ..., item49], [item50, ..., item99], ...]
```

**Use Cases**:
- Batch database inserts/updates from streaming data
- Chunked API requests to respect payload size limits
- Processing large datasets in fixed-size batches
- Grouping streaming data for parallel processing
- Splitting file reads into manageable blocks

**Key Features**:
- Pure transformation (yields chunks as lists)
- Handles partial final chunks gracefully
- Validates chunk size (must be >= 1)
- Works with any async iterable
- Lazy evaluation (chunks are generated on-demand)

**Difference from `batched_async_iterable`**:
- `async_chunks`: Pure transformation → yields chunks
- `batched_async_iterable`: Processing → applies function to batches

**Files**: `src/nodetool/concurrency/async_iterators.py`

**Date**: 2026-02-16
