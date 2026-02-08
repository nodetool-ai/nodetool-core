# Async File System Operations Pattern

**Insight**: When working with file system operations in async functions, always use `asyncio.to_thread()` for blocking calls like `os.makedirs()`, `os.path.exists()`, etc.

**Rationale**: Python's `os` module functions are synchronous and block the event loop. In an async workflow engine where multiple tasks may be executing concurrently, blocking calls can significantly degrade performance. Using `asyncio.to_thread()` offloads the blocking operation to a separate thread, allowing the event loop to continue processing other tasks.

**Example**:
```python
# Before (blocking):
async def process(self, context, params):
    parent_dir = os.path.dirname(output_file)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)  # Blocks event loop!

# After (non-blocking):
async def process(self, context, params):
    parent_dir = os.path.dirname(output_file)
    if parent_dir:
        await asyncio.to_thread(os.makedirs, parent_dir, exist_ok=True)  # Non-blocking!
```

**Impact**: Measurable performance improvement in concurrent workflow execution, especially when multiple file operations occur simultaneously.

**Files**:
- `src/nodetool/agents/tools/pdf_tools.py`
- `src/nodetool/agents/tools/http_tools.py`
- `src/nodetool/agents/tools/workspace_tools.py`

**Date**: 2026-02-08
