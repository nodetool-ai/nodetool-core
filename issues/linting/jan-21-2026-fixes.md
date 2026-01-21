# Linting Issues Fixed - January 21, 2026

**Problem**: Three linting violations were reported by ruff across the codebase.

**Solution**: Fixed all violations in the following files:

1. `src/nodetool/workflows/event_logger.py:76`:
   - Removed trailing whitespace from blank line (W293)

2. `src/nodetool/workflows/event_logger.py:109`:
   - Replaced `asyncio.TimeoutError` with builtin `TimeoutError` (UP041)

3. `src/nodetool/workflows/workflow_runner.py:969`:
   - Removed f-string prefix from `log.info(f"EventLogger stopped for suspension")` (F541)

**Additional Changes**:
- `src/nodetool/workflows/workflow_runner.py:919`: Reformatted long line to multi-line for readability (black formatting)

**Why**: These linting violations indicate code quality issues. Fixing them improves code maintainability and passes the project's CI checks.

**Files Modified**:
- `src/nodetool/workflows/event_logger.py`
- `src/nodetool/workflows/workflow_runner.py`

**Date**: 2026-01-21
