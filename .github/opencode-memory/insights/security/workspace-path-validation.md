# Workspace Path Validation Pattern

**Insight**: The codebase has a robust `resolve_workspace_path()` function in `src/nodetool/io/path_utils.py` that prevents path traversal attacks by validating resolved paths stay within the workspace directory. All file operations should use this pattern instead of raw `os.path.abspath()`.

**Rationale**: Agent tools that access the filesystem need to operate within a bounded "sandbox" directory. Using `os.path.abspath()` alone allows path traversal via `../` sequences. The `resolve_workspace_path()` function:
1. Normalizes paths relative to workspace
2. Resolves them to absolute paths
3. Validates the result is still within the workspace directory using `os.path.commonprefix()`
4. Raises `ValueError` if path traversal is detected

**Example**:
\`\`\`python
# VULNERABLE:
full_path = os.path.abspath("../../../etc/passwd")  # Escapes workspace!

# SECURE:
full_path = context.resolve_workspace_path("../../../etc/passwd")  # Raises ValueError
\`\`\`

**Impact**: Provides a foundational security primitive for agent file operations, preventing unauthorized file access.

**Files**:
- `src/nodetool/io/path_utils.py` (implementation)
- `src/nodetool/workflows/processing_context.py` (ProcessingContext.resolve_workspace_path)
- `src/nodetool/agents/tools/filesystem_tools.py` (usage)

**Date**: 2026-02-11
