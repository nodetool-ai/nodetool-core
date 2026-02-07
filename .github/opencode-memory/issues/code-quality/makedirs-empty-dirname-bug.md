# os.makedirs() with Empty dirname Argument Bug

**Problem**: Multiple files use `os.makedirs(os.path.dirname(path), exist_ok=True)` which fails when `path` has no directory component (e.g., "file.txt"). In this case, `os.path.dirname()` returns an empty string, and `os.makedirs("")` raises an error.

**Solution**: Added check for empty dirname before calling `os.makedirs()`:
```python
parent_dir = os.path.dirname(full_path)
if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)
```

**Why**: When saving files with just a filename (no directory path), the code would crash with "FileNotFoundError: [Errno 2] No such file or directory: ''". This fix allows files to be saved directly in the current/workspace directory without requiring a subdirectory.

**Files**:
- `src/nodetool/agents/tools/browser_tools.py` (3 locations)
- `src/nodetool/agents/tools/filesystem_tools.py` (1 location)
- `src/nodetool/agents/tools/http_tools.py` (1 location)
- `src/nodetool/agents/tools/pdf_tools.py` (3 locations)
- `src/nodetool/agents/tools/workspace_tools.py` (1 location)

**Date**: 2026-02-07
