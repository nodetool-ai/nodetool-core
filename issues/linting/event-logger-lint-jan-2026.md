# Event Logger Linting Issues (Jan 2026)

**Problem**: Linting errors in `src/nodetool/workflows/event_logger.py`:
- W293: Blank line contains whitespace at line 76
- UP041: Using `asyncio.TimeoutError` instead of builtin `TimeoutError`

**Solution**: 
- Removed whitespace from blank line after docstring
- Replaced `asyncio.TimeoutError` with `TimeoutError`

**Files**: 
- `src/nodetool/workflows/event_logger.py:76`
- `src/nodetool/workflows/event_logger.py:109`

**Date**: 2026-01-21
