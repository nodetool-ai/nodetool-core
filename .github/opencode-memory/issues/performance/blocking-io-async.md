# Blocking I/O in Async Functions

**Problem**: Several async functions were using synchronous `open()` calls, blocking the event loop during file I/O operations.

**Files Fixed**:
- `src/nodetool/workflows/processing_context.py:1105, 1143` - `download_file()` method
- `src/nodetool/workflows/processing_context.py:2058` - `video_from_numpy()` method  
- `src/nodetool/agents/task_planner.py:293` - `_load_existing_plan()` method

**Solution**: Use `asyncio.to_thread()` for blocking file reads and `Path.read_bytes()` for async file content retrieval.

**Impact**: Prevents event loop blocking during file operations, improving responsiveness in async contexts.

**Date**: 2026-01-16
