# Print Statements in Production Code

**Problem**: Production code in `src/nodetool/providers/comfy_api.py` used `print()` statements instead of proper logging.

**Solution**: Replaced `print()` calls with `logger.error()` and `logger.warning()` calls.

**Files**: 
- `src/nodetool/providers/comfy_api.py` (lines 566, 569)

**Date**: 2026-01-19
