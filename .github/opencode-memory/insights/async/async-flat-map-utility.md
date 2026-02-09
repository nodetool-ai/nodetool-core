# async_flat_map Utility

**Insight**: Flat mapping is a common pattern for one-to-many transformations on async iterables, combining mapping and flattening into a single operation.

**Rationale**: When working with async iterables, it's common to need to transform each element into multiple elements (e.g., splitting a string into characters, expanding a range, or processing nested data structures). The `async_flat_map` utility provides a composable way to handle this pattern without intermediate list allocations.

**Example**:
```python
from nodetool.concurrency import async_flat_map, async_list

# Split each number into a range
async def split(x):
    for j in range(x):
        yield j

async def gen():
    for i in range(3):
        yield i

result = await async_list(async_flat_map(split, gen()))
# Result: [0, 0, 1] (split(0)=[], split(1)=[0], split(2)=[0, 1])

# Flatten nested lists
async def flatten(lst):
    for item in lst:
        yield item

async def nested_gen():
    for lst in [[1, 2], [3, 4, 5], [6]]:
        yield lst

result = await async_list(async_flat_map(flatten, nested_gen()))
# Result: [1, 2, 3, 4, 5, 6]
```

**Impact**: Reduces code complexity for common async iterable transformations and provides a consistent API with other async iterator utilities like `async_map` and `async_filter`.

**Files**:
- `src/nodetool/concurrency/async_iterators.py`: Implementation
- `src/nodetool/concurrency/__init__.py`: Export
- `tests/concurrency/test_async_iterators.py`: Test coverage (13 tests)

**Date**: 2026-02-09
