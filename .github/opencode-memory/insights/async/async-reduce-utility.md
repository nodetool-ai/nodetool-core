# Async Reduce Utility

**Insight**: Added `async_reduce` function for reducing async iterables to a single value using a reduction function.

**Rationale**: Python's built-in `functools.reduce` doesn't support async iterables or async reduction functions. The `async_reduce` utility fills this gap, providing a functional programming primitive for aggregating async sequences. This is particularly useful for computing sums, products, building collections, or any cumulative operation on async data streams.

**Example**:

```python
from nodetool.concurrency import async_reduce

# Sum all numbers from an async generator
total = await async_reduce(lambda acc, x: acc + x, async_generator(), 0)

# Multiply all values
product = await async_reduce(lambda acc, x: acc * x, numbers_gen(), 1)

# Build a list from items
async def append(acc, x):
    acc.append(x)
    return acc
result_list = await async_reduce(append, items_gen(), [])

# Use async reduction function
async def async_sum(acc, x):
    await some_async_operation()
    return acc + x
total = await async_reduce(async_sum, data_gen(), 0)

# Find maximum value
max_val = await async_reduce(lambda acc, x: acc if acc > x else x, numbers_gen(), 0)
```

**Impact**: Completes the functional programming toolkit for async iterables alongside `async_map`, `async_filter`, etc. Supports both sync and async reduction functions with automatic coroutine detection. Fully type-hinted with Generic TypeVars for type safety. Includes 14 comprehensive tests covering edge cases like empty iterables, single items, various accumulator types (list, dict, tuple, set), and complex transformations.

**Files**:
- `src/nodetool/concurrency/async_iterators.py` - Added `async_reduce` function
- `src/nodetool/concurrency/__init__.py` - Updated exports
- `tests/concurrency/test_async_iterators.py` - Added `TestAsyncReduce` class with 14 tests

**Date**: 2026-02-09
