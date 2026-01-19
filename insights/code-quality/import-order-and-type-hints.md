# Import Order and Type Hint Best Practices

**Insight**: Maintaining proper import order and modern type hints improves code quality and reduces linting violations.

**Rationale**:
1. **Import Order**: Python's style guide (PEP 8) specifies that imports should be at the top of files in a specific order: standard library, third-party, local application. This prevents subtle bugs from import-time side effects and makes dependencies clear.

2. **Type Hints**: Python 3.9+ supports using built-in collection types directly as type hints (`list[str]` instead of `List[str]`). This is more readable and is the recommended style.

**Example**:

```python
# Bad (E402 - imports not at top)
import logging
log = logging.getLogger(__name__)
from nodetool.workflows.processing_context import ProcessingContext

# Bad (UP006 - deprecated List syntax)
from typing import List, Dict
def process(items: List[str]) -> Dict[str, int]:
    ...

# Good
import logging
from typing import List, Dict

log = logging.getLogger(__name__)
from nodetool.workflows.processing_context import ProcessingContext

# Modern (Python 3.9+)
def process(items: list[str]) -> dict[str, int]:
    ...
```

**Impact**:
- Fewer linting violations
- Better code organization
- Improved readability
- Future-proof code

**Files**: All Python files in `src/nodetool/`

**Date**: 2026-01-19
