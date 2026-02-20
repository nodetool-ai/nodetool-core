# Linting Fixes: Import Order and Whitespace

**Insight**: Code quality tools (ruff) automatically catch and can fix many formatting issues including import ordering and trailing whitespace.

**Rationale**: Consistent code formatting improves readability and reduces merge conflicts. Using automated linters ensures consistency across the codebase without manual review.

**Example**:
```python
# Before: imports not properly sorted, trailing whitespace on blank lines
from contextlib import contextmanager

import asyncio


def example():
    pass  # Line with trailing whitespace


# After: ruff --fix sorts imports and removes trailing whitespace
import asyncio
from contextlib import contextmanager, suppress


def example():
    pass
```

**Impact**: The lint fixes in this PR resolved:
- I001 (Import block is un-sorted or un-formatted) in `actor.py`
- W291/W293 (Trailing/blank line whitespace) in `memory_utils.py`
- RUF022 (`__all__` is not sorted) in `memory_utils.py`
- F841 (Unused local variables) in `memory_utils.py`

**Files**:
- `src/nodetool/workflows/actor.py`
- `src/nodetool/workflows/memory_utils.py`

**Date**: 2026-02-20
