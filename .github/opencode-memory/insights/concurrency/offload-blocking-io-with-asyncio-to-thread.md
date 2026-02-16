# Offloading Blocking I/O with asyncio.to_thread

**Insight**: Use `await asyncio.to_thread()` for blocking I/O operations in async functions to prevent event loop blocking.

**Rationale**:
- Blocking operations in async functions (like `file.read()`) prevent the event loop from processing other tasks
- `asyncio.to_thread()` offloads synchronous functions to a thread pool, keeping the event loop responsive
- Essential for file I/O, network I/O without async support, and CPU-bound operations
- Thread pool size is limited, so use sparingly for truly blocking operations only

**Example**:
```python
# Bad - blocks event loop
async def upload_async(file: IO) -> None:
    async with aiofiles.open("output.bin", "wb") as f:
        chunk = file.read(1024 * 1024)  # BLOCKING - prevents other tasks from running
        await f.write(chunk)

# Good - offloads blocking read to thread pool
async def upload_async(file: IO) -> None:
    async with aiofiles.open("output.bin", "wb") as f:
        chunk = await asyncio.to_thread(file.read, 1024 * 1024)  # Non-blocking
        await f.write(chunk)
```

**Impact**: Improved throughput and responsiveness in async applications, especially under load.

**Files**: `src/nodetool/storage/file_storage.py`

**Date**: 2026-02-16
