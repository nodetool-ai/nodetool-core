# AsyncByteStream Test Coverage

**Insight**: Added comprehensive test coverage for `src/nodetool/concurrency/async_iterators.py` module which was previously undertested despite being listed as a core feature in `features.md`.

**Rationale**: The `AsyncByteStream` class is used for async byte sequence iteration in chunks, which is important for processing large data streams efficiently. Having tests ensures correct behavior and prevents regressions.

**Coverage Added**:
- Constructor tests (default and custom chunk sizes)
- `__aiter__` method tests
- `__anext__` method tests with various data sizes
- Edge cases: empty data, single chunk, exact multiples, remainders
- Boundary conditions: chunk_size=1, chunk_size larger than data
- Data types: binary data with null bytes, UTF-8 encoded unicode
- `StopAsyncIteration` behavior at end of data
- Index tracking during iteration

**Test Count**: 19 new tests covering all public APIs

**Files**: `tests/concurrency/test_async_iterators.py`

**Date**: 2026-01-17
