# Path Traversal in Agent Filesystem Tools

**Problem**: The `WriteFileTool`, `ReadFileTool`, and `ListDirectoryTool` in `filesystem_tools.py` used `os.path.abspath()` on user-supplied paths without validating they stayed within the workspace directory.

**Solution**: Updated all three tools to use `context.resolve_workspace_path()` instead of `os.path.abspath()`. This function validates that resolved paths stay within the configured workspace directory and raises an error if path traversal is attempted.

**Why**: An agent could read or write arbitrary files on the host system by using path traversal sequences like `../../etc/passwd` or `../../../../../../root/.ssh/id_rsa`.

**Files**:
- `src/nodetool/agents/tools/filesystem_tools.py` (WriteFileTool, ReadFileTool, ListDirectoryTool)

**Date**: 2026-02-11
