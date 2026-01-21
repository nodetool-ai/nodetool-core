# Linting Issues Fixed Jan 2026

**Problem**: Three linting issues were found:
1. `W293`: Blank line contains whitespace in `src/nodetool/workflows/event_logger.py:76`
2. `UP041`: Aliased `asyncio.TimeoutError` should use builtin `TimeoutError` at `src/nodetool/workflows/event_logger.py:109`
3. `F541`: f-string without placeholders at `src/nodetool/workflows/workflow_runner.py:969`

**Solution**:
1. Removed trailing whitespace from blank line
2. Changed `except asyncio.TimeoutError:` to `except TimeoutError:`
3. Removed `f` prefix from `f"EventLogger stopped for suspension"`

**Files**:
- `src/nodetool/workflows/event_logger.py`
- `src/nodetool/workflows/workflow_runner.py`

**Date**: 2026-01-21
