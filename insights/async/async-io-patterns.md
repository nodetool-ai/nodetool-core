# Async I/O Patterns in Python

**Insight**: Use `asyncio.to_thread()` for blocking I/O operations in async functions to prevent event loop blocking.

**Rationale**: Async functions should never block the event loop. When you need to perform blocking I/O (file system, network, etc.), you must either:
1. Use async-native libraries (e.g., `aiofiles` for file I/O)
2. Offload blocking operations to a thread pool using `asyncio.to_thread()`

**Best Practices**:
1. **File Operations**: Use `aiofiles.open()` instead of `open()`
2. **File System Checks**: Use `await asyncio.to_thread(os.path.exists, path)` instead of `os.path.exists(path)`
3. **Complex Blocking Operations**: Extract to a sync helper function, then call via `asyncio.to_thread()`

**Example - File Existence Check**:
```python
# BAD - Blocks event loop
file_existed = os.path.exists(full_path)

# GOOD - Offloaded to thread pool
file_existed = await asyncio.to_thread(os.path.exists, full_path)
```

**Example - Complex Directory Scan**:
```python
# BAD - Blocks event loop for entire scan
async def calculate_cache_size(cache_dir: str):
    total_size = 0
    if os.path.exists(cache_dir):
        for dirpath, _dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    return total_size

# GOOD - Extract sync helper, offload to thread
def _calculate_cache_size_sync(cache_dir: str) -> tuple[int, str | None]:
    total_size = 0
    error = None
    try:
        if os.path.exists(cache_dir):
            for dirpath, _dirnames, filenames in os.walk(cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
    except Exception as e:
        error = str(e)
    return total_size, error

async def calculate_cache_size(cache_dir: str):
    total_size, error = await asyncio.to_thread(_calculate_cache_size_sync, cache_dir)
    if error:
        raise Exception(error)
    return total_size
```

**Impact**: Prevents event loop blocking, improves responsiveness, allows other async tasks to run during I/O operations.

**Files**: See related issues in `issues/code-quality/blocking-file-ops-in-async-functions.md`

**Date**: 2026-02-18
