# AsyncChannel send_nowait/receive_nowait Incorrectly Marked as Async

**Problem**: `AsyncChannel.send_nowait()` and `AsyncChannel.receive_nowait()` were marked as `async def` but wrapped synchronous queue operations (`put_nowait()`, `get_nowait()`), causing them to return coroutines instead of actual values.

**Solution**: Changed both methods from `async def` to regular `def` since they don't perform any async operations.

**Why**: The underlying asyncio.Queue methods `put_nowait()` and `get_nowait()` are synchronous operations that don't require await. Marking the wrapper methods as async caused them to return coroutine objects instead of the actual values, making the API inconsistent with the expected behavior.

**Files**: `src/nodetool/concurrency/async_channel.py`

**Date**: 2026-02-08
