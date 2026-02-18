# Blocking File Operations in Async Functions

**Problem**: Several async functions were using blocking file system operations (`os.path.exists()`, `os.walk()`, `os.path.getsize()`) without using `asyncio.to_thread()`, which blocks the event loop during I/O operations.

**Solution**:
1. `workspace_tools.py`: Moved `os.path.exists()` check before file write and wrapped it with `asyncio.to_thread()`
2. `admin_operations.py`: Refactored `calculate_cache_size()` to use a synchronous helper function called via `asyncio.to_thread()`

**Why**: In async code, blocking I/O operations can stall the entire event loop, preventing other async tasks from running. The async version using `asyncio.to_thread()` allows the event loop to continue processing other tasks while the I/O happens in the background.

**Files**:
- `src/nodetool/agents/tools/workspace_tools.py`
- `src/nodetool/deploy/admin_operations.py`

**Impact**:
- Improved async performance by eliminating blocking I/O in event loop
- Better responsiveness for concurrent operations
- All existing tests continue to pass

**Date**: 2026-02-18
