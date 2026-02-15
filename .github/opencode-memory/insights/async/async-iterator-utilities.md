# Async Iterator Utilities

**Insight**: Added utility functions for working with async iterables in `src/nodetool/concurrency/async_iterators.py`: `async_take`, `async_slice`, `async_first`, `async_list`, and `async_merge`.

**Rationale**: While Python 3.11+ has excellent async iterator support, common operations like taking the first N items, slicing, or consuming to a list require boilerplate code. These utilities provide familiar functional programming patterns for async iterables, making async code more readable and maintainable.

**Example**:

```python
from nodetool.concurrency import async_take, async_first, async_list, async_merge

# Take first N items from an async generator
first_5 = await async_take(async_generator(), 5)

# Get just the first item (with optional default)
first = await async_first(async_generator(), default=None)

# Slice async iterables like lists
items_5_to_10 = await async_slice(async_generator(), 5, 10)

# Consume to list
all_items = await async_list(async_generator())

# Merge multiple async iterables sequentially
combined = await async_list(async_merge(gen1(), gen2(), gen3()))
```

**Impact**: Provides reusable async iterator utilities that follow the codebase's async-first patterns. Reduces boilerplate code when working with async generators and iterables. All utilities are fully type-hinted with Generic TypeVar support for better IDE autocomplete and type checking.

**Files**:
- `src/nodetool/concurrency/async_iterators.py` - Added utility functions
- `src/nodetool/concurrency/__init__.py` - Updated exports
- `tests/concurrency/test_async_iterators.py` - 29 new tests covering all utilities

**Date**: 2026-02-08
