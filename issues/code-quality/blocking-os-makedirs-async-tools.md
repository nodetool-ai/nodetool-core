# Blocking os.makedirs() in Async Tool Functions

**Problem**: 3 async tool functions were using blocking `os.makedirs()` calls instead of wrapping them with `asyncio.to_thread()`. This blocks the event loop, degrading performance in an async workflow engine.

**Solution**: Wrapped all `os.makedirs()` calls in async functions with `await asyncio.to_thread(os.makedirs, ...)` across 3 files:

1. `src/nodetool/agents/tools/pdf_tools.py` - 3 `os.makedirs()` calls fixed
2. `src/nodetool/agents/tools/http_tools.py` - 1 `os.makedirs()` call fixed
3. `src/nodetool/agents/tools/workspace_tools.py` - 1 `os.makedirs()` call fixed

**Impact**:
- Improved async performance by eliminating blocking I/O in event loop
- All 5 ASYNC230 violations in these files resolved
- Tests pass (pre-existing failures unrelated to this fix)
- Lint passes
- Typecheck passes

**Files Modified**:
- `src/nodetool/agents/tools/pdf_tools.py` - Added `asyncio` import, wrapped `os.makedirs()` calls
- `src/nodetool/agents/tools/http_tools.py` - Added `asyncio` import, wrapped `os.makedirs()` call
- `src/nodetool/agents/tools/workspace_tools.py` - Added `asyncio` import, wrapped `os.makedirs()` call

**Date**: 2026-02-08
