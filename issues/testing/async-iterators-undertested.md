# AsyncByteStream Was Undertested

**Problem**: The `AsyncByteStream` class in `src/nodetool/concurrency/async_iterators.py` was listed as a core feature in `features.md` but had only a single basic test in `tests/common/test_extra_utils.py`.

**Solution**: Created `tests/concurrency/test_async_iterators.py` with 19 comprehensive tests covering all public APIs and edge cases.

**Why**: Async byte stream iteration is important for processing large data efficiently. Tests ensure correct behavior and prevent regressions when the codebase evolves.

**Files**:
- `src/nodetool/concurrency/async_iterators.py` - The module being tested
- `tests/concurrency/test_async_iterators.py` - New comprehensive test file

**Date**: 2026-01-17
