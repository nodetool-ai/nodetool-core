# Unused Parameter in _check_depends_on

**Problem**: The `_check_depends_on` method in `task_executor.py` had an unused `workspace_dir` parameter that was never used in the function body. The parameter was typed as `str` but could receive `str | None` values, causing type errors.

**Solution**: Removed the unused `workspace_dir` parameter from the function signature and updated the call site. Also fixed the docstring to accurately describe the function's behavior (checking task dependencies, not file dependencies).

**Files**:
- `src/nodetool/agents/task_executor.py:203-212`

**Date**: 2026-01-14
