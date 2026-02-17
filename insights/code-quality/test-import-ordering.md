# Test File Import Ordering

**Insight**: Imports in test files should follow the same ordering standards as production code - standard library first, then third-party, then local imports.

**Rationale**:
- Consistent import order improves code readability
- Helps identify dependencies at a glance
- Ruff's I001 rule enforces this automatically
- Using `ruff check --fix` can auto-correct import ordering

**Example**:
```python
# Problem: Local import before standard library
from nodetool.workflows.workflow_runner import WorkflowRunner
from unittest.mock import MagicMock

# Solution: Standard library first
from unittest.mock import MagicMock

from nodetool.workflows.workflow_runner import WorkflowRunner
```

**Impact**: Fixed 9 import ordering issues in `tests/workflows/test_control_edges.py`

**Files**:
- `tests/workflows/test_control_edges.py` (lines 924-926, 972-973, 1014-1016, 1065-1066, 1190-1191, 1225-1226, 1267-1268)

**Date**: 2026-02-17
