# Concurrency Module Typing Modernization

**Insight**: Modernized typing imports in the concurrency module to use Python 3.9+ syntax and `collections.abc` for better compatibility with Python 3.11+.

**Changes Made**:
- `src/nodetool/concurrency/async_utils.py`:
  - Removed `from typing import Optional`
  - Changed `Optional[float]` to `float | None` in function signature

- `src/nodetool/concurrency/batching.py`:
  - Changed `from typing import AsyncIterator, Awaitable, Callable, TypeVar`
  - To `from collections.abc import AsyncIterator, Awaitable, Callable` and `from typing import TypeVar`

- `src/nodetool/concurrency/retry.py`:
  - Changed `from typing import Any, Callable, TypeVar`
  - To `from collections.abc import Callable` and `from typing import Any, TypeVar`

**Rationale**:
- `collections.abc` is the canonical location for abstract base classes like `AsyncIterator`, `Awaitable`, and `Callable`
- Using `| None` syntax is more idiomatic and reduces reliance on `typing.Optional`
- Aligns with the existing "typing modernization" effort documented in the codebase

**Files**:
- `src/nodetool/concurrency/async_utils.py`
- `src/nodetool/concurrency/batching.py`
- `src/nodetool/concurrency/retry.py`

**Date**: 2026-01-17
