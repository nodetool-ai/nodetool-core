# Missing None Check for workspace_dir

**Problem**: The `ChangeToWorkspaceCommand.execute()` method in `workspace.py` accessed `cli.context.workspace_dir` without checking if it was None, causing type errors when Path() was called on a None value.

**Solution**: Added a None check at the start of the execute method that returns an error message if workspace_dir is not set.

**Files**:
- `src/nodetool/chat/commands/workspace.py:18-19`

**Date**: 2026-01-14
