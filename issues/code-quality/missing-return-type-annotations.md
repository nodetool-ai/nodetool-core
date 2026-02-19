# Missing Return Type Annotations

**Problem**: Many async functions and methods in the codebase lack explicit return type annotations, particularly for functions that return `None`.

**Solution**: Add explicit `-> None` return type annotations to functions that don't return values.

**Why**: Explicit return type annotations:
- Improve type safety by allowing type checkers to verify correct usage
- Make the code's intent more clear to readers
- Align with Python 3.11+ best practices for type annotations
- Help catch bugs where return values are incorrectly used

**Files**: Many files across the codebase, particularly in `src/nodetool/chat/`, `src/nodetool/agents/`, and `src/nodetool/api/`

**Date**: 2026-02-19

**Status**: Partially fixed - chat module functions updated, but ~2500+ type errors remain
