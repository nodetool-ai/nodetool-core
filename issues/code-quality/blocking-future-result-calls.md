# Blocking `.result()` Calls in Async Contexts

**Problem**: The `run_workflow()` function in `src/nodetool/workflows/run_workflow.py` was calling `.result()` on `concurrent.futures.Future` objects inside async generators. This blocks the event loop since `.result()` waits synchronously for the future to complete.

**Solution**: Modified the code to:
1. Check if the future is done before calling `.result()`
2. Use `asyncio.wrap_future()` to convert the concurrent future to an awaitable
3. Use `asyncio.wait_for()` with a timeout to properly await the future
4. Properly handle `CancelledError` and other exceptions

**Why**: Blocking calls in async code can cause event loop stalls, especially in workflows that take longer than expected. The original code assumed futures were always done, but this isn't guaranteed.

**Files**:
- `src/nodetool/workflows/run_workflow.py`

**Date**: 2026-01-15
