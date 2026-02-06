# Parallel Map Utility

**Insight**: Added `parallel_map` for parallel processing of individual items with controlled concurrency.

**Rationale**: The existing `process_in_batches` utility processes items in groups, while `parallel_map` processes individual items concurrently. This is useful for API calls, file operations, or any operation where you want item-level parallelism without batching logic. The function complements `gather_with_concurrency` by providing ordered results and a simpler interface.

**Example**:
```python
from nodetool.concurrency import parallel_map

# Fetch multiple URLs concurrently
responses = await parallel_map(
    items=["https://a.com", "https://b.com", "https://c.com"],
    mapper=fetch_url,
    max_concurrent=10,
)
# Returns responses in same order as input URLs

# Process files in parallel
results = await parallel_map(
    items=file_paths,
    mapper=process_file,
    max_concurrent=5,
)
```

**Impact**: Simplifies concurrent item processing patterns that were previously implemented ad-hoc. Useful for batch API calls, file processing, and any parallel data transformation.

**Files**:
- `src/nodetool/concurrency/parallel_map.py`
- `tests/concurrency/test_parallel_map.py`

**Date**: 2026-01-21
