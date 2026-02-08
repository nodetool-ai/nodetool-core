# Blocking shutil.rmtree in Async Function

**Problem**: `shutil.rmtree()` is a blocking I/O operation called directly in an async function without using `asyncio.to_thread()`, which blocks the event loop during directory deletion.

**Solution**: Replace `shutil.rmtree(tmp_dir)` with `await asyncio.to_thread(shutil.rmtree, tmp_dir)` to offload the blocking operation to a thread pool.

**Why**: In async code, blocking I/O operations like directory tree removal can stall the entire event loop, preventing other async tasks from running. The async version using `asyncio.to_thread()` allows the event loop to continue processing other tasks while the directory deletion happens in the background.

**Files**: `src/nodetool/deploy/collection_routes.py`

**Date**: 2026-02-08
