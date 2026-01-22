# Linting Issues Fixed - January 2026

**Problem**: Linting checks were failing with 3 issues:
1. W293: Blank line contains whitespace in `event_logger.py`
2. UP041: Using aliased `asyncio.TimeoutError` instead of builtin `TimeoutError`
3. F541: f-string without any placeholders in `workflow_runner.py`

**Solution**: 
- Removed trailing whitespace on blank line in `src/nodetool/workflows/event_logger.py:76`
- Replaced `asyncio.TimeoutError` with builtin `TimeoutError` in `src/nodetool/workflows/event_logger.py:109`
- Removed f-string prefix from static string in `src/nodetool/workflows/workflow_runner.py:969`

**Files**: 
- `src/nodetool/workflows/event_logger.py`
- `src/nodetool/workflows/workflow_runner.py`

**Date**: 2026-01-21
