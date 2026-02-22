# Async Partition and Chunked Utilities

**Insight**: Added `async_partition` and `async_chunked` utilities to `src/nodetool/concurrency/async_iterators.py` for async iterable manipulation.

**Rationale**: The existing async iterator utilities were comprehensive but lacked two common patterns:
1. **Partition**: Splitting items into two groups (pass/fail, valid/invalid) based on a predicate
2. **Chunked**: Grouping items into fixed-size batches for batch processing or rate limiting

These utilities complement the existing `async_filter`, `async_map`, `async_take`, and `async_slice` functions, providing a more complete toolkit for async iterable manipulation.

**Example**:

```python
from nodetool.concurrency import async_partition, async_chunked

# Partition numbers into even and odd
async def gen_numbers():
    for i in range(10):
        yield i

evens, odds = await async_partition(lambda x: x % 2 == 0, gen_numbers())
# evens = [0, 2, 4, 6, 8]
# odds = [1, 3, 5, 7, 9]

# Chunk items into groups of 3
async for chunk in async_chunked(gen_numbers(), 3):
    await process_batch(chunk)
    # chunks = [0, 1, 2], [3, 4, 5], [6, 7, 8], [9]

# Works with async predicates too
async def is_valid(item):
    return await validate(item)

valid, invalid = await async_partition(is_valid, items_stream())
```

**Features**:
- **async_partition**: Returns tuple of (matching_items, non_matching_items)
  - Supports both sync and async predicates
  - Preserves original order within each group
  - Useful for validation, filtering with both branches, classification

- **async_chunked**: Yields lists of up to chunk_size items
  - Last chunk may be smaller if total items not divisible by chunk_size
  - Validates chunk_size > 0
  - Useful for batch processing, rate limiting, pagination

**Impact**: Reduces code duplication for common patterns like splitting validation results or batching API calls. Provides consistent, well-tested utilities that work with both sync and async functions.

**Files**:
- `src/nodetool/concurrency/async_iterators.py`
- `src/nodetool/concurrency/__init__.py`
- `tests/concurrency/test_async_iterators.py`

**Date**: 2026-02-20
