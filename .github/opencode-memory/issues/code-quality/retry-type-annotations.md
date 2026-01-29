# Type Annotation Improvements in retry.py

**Problem**: The `retry.py` module had several type safety issues:
1. Unreachable code after a `while True` loop with `raise` at the end
2. Missing type annotations on class attributes
3. Overly broad `Any` return type in function signatures
4. Unused `last_exception` variable

**Solution**:
1. Removed unreachable code after the retry loop
2. Added explicit type annotations to all `RetryPolicy` class attributes
3. Updated function signatures to use `Coroutine[Any, Any, T]` for proper async type checking
4. Removed unused `last_exception` variable

**Changes**:
- `src/nodetool/concurrency/retry.py`:
  - Added type annotations: `self.max_retries: int`, `self.initial_delay: float`, etc.
  - Changed `func: Callable[[], Any]` to `func: Callable[[], Coroutine[Any, Any, T]]`
  - Changed return types from `Any` to proper generic `T` type
  - Removed unreachable `raise last_exception` statement

**Date**: 2026-01-15
