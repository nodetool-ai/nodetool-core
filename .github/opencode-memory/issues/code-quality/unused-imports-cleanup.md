# Unused Imports Cleanup

**Problem**: Several files had unused imports causing linting failures (F401 errors).

**Solution**: Removed unused imports from 5 files:
- `src/nodetool/workflows/workflow_runner.py`: Removed `logging` and `contextlib.suppress`
- `src/nodetool/agents/tools/serp_tools.py`: Removed `Environment` import
- `src/nodetool/agents/tools/tool_registry.py`: Removed `typing.Any` import
- `src/nodetool/api/asset.py`: Removed `Union` import
- `src/nodetool/api/file.py`: Removed `timezone` and `Workspace` imports

**Why**: Unused imports clutter the codebase, increase load times, and indicate incomplete refactoring. The codebase follows the pattern of keeping imports minimal and focused.

**Files**: 
- src/nodetool/workflows/workflow_runner.py
- src/nodetool/agents/tools/serp_tools.py
- src/nodetool/agents/tools/tool_registry.py
- src/nodetool/api/asset.py
- src/nodetool/api/file.py

**Date**: 2026-01-17
