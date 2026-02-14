# Missing Exception Chaining in User Management

**Problem**: Exception handlers in `users.py` raised new HTTPException instances without chaining to the original exception, losing stack trace context.

**Solution**: Added `from e` exception chaining to all except blocks that re-raise as HTTPException.

**Why**: Proper exception chaining preserves the original traceback and makes debugging easier by showing the full error context.

**Files**: `src/nodetool/api/users.py`

**Date**: 2026-02-14
