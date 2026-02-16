# Blocking I/O in Async Functions

**Problem**: The `upload()` and `upload_sync()` functions in `file_storage.py` used synchronous `content.read(1024 * 1024)` calls inside async functions, blocking the event loop during file reads.

**Solution**: Wrapped the `content.read()` calls with `await asyncio.to_thread()` to offload the blocking I/O operation to a thread pool, preventing event loop blocking.

**Why**: In async code, blocking I/O operations prevent the event loop from processing other tasks, reducing throughput and causing performance issues. Using `asyncio.to_thread()` ensures the event loop remains responsive while the I/O operation executes in a background thread.

**Files**:
- `src/nodetool/storage/file_storage.py:62, 75`

**Date**: 2026-02-16
