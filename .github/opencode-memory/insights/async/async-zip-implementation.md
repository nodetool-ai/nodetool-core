# async_zip Implementation

**Insight**: When implementing async_zip, the function must stop iteration when the shortest iterable is exhausted, mirroring Python's built-in zip() behavior.

**Rationale**:
The built-in zip() function stops at the shortest iterable, which is the expected Python behavior. An async version must maintain this semantic while handling multiple async iterators simultaneously.

**Example**:
```python
async def async_zip(*iterables: AsyncIterable[T]) -> AsyncIterator[tuple[T, ...]]:
    """
    Combine multiple async iterables into tuples of elements.

    Yields tuples containing the i-th element from each of the argument
    iterables. Stops when the shortest iterable is exhausted.
    """
    iterators = [ait.__aiter__() for ait in iterables]

    if not iterators:
        return

    while True:
        try:
            items = []
            for it in iterators:
                item = await it.__anext__()
                items.append(item)
            yield tuple(items)
        except StopAsyncIteration:
            # Any iterator exhausted - stop iteration
            break
```

**Key Design Decisions**:
1. Convert all iterables to iterators upfront using `__aiter__()`
2. Try to get the next item from each iterator in sequence
3. If any iterator raises `StopAsyncIteration`, stop the entire zip operation
4. Handle edge cases: no iterables, single iterable, empty iterables
5. Support any number of iterables via `*iterables`

**Testing Requirements**:
- Equal length iterables
- Different length iterables (stops at shortest)
- Three or more iterables
- Empty iterables
- Single iterable
- No iterables
- Different types in same zip
- None and Falsey values
- Order preservation

**Impact**:
Completes the async iterator utility suite, providing parity with Python's built-in zip() for async workflows. Essential for combining data from multiple async sources (e.g., streaming from multiple APIs, processing paired data streams).

**Files**:
- `src/nodetool/concurrency/async_iterators.py`
- `tests/concurrency/test_async_iterators.py`

**Date**: 2026-02-10
