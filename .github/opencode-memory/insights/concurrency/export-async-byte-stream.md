# Exporting Existing Async Utilities

**Insight**: Always export all public utilities defined in module files via `__init__.py` to make them discoverable and usable by consumers of the package.

**Rationale**: The `AsyncByteStream` class was implemented in `async_iterators.py` but was not exported in `__init__.py`, making it invisible to users importing from `nodetool.concurrency`.

**Example**: Adding the export:
```python
# In src/nodetool/concurrency/__init__.py
from .async_iterators import AsyncByteStream

__all__ = [
    "AsyncByteStream",
    # ... other exports
]
```

**Impact**: Users can now import `AsyncByteStream` using `from nodetool.concurrency import AsyncByteStream`.

**Files**: `src/nodetool/concurrency/__init__.py`, `src/nodetool/concurrency/async_iterators.py`

**Date**: 2026-01-14
