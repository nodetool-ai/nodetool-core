# Type Guard Best Practice with `typing.cast`

**Insight**: When using `hasattr` for dynamic method checks, the type checker doesn't automatically narrow the type. Using `typing.cast()` combined with a runtime `hasattr` check properly narrows types for static analysis.

**Example**:
```python
from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
    from some_module import SpecificType

def process_node(node: BaseNode) -> bool:
    if not hasattr(node, "_set_resuming_state"):
        return False
    # This won't work - type checker doesn't narrow
    node._set_resuming_state(state, 0)
    
    # This works - explicit cast tells type checker
    specific = cast("SpecificType", node)
    specific._set_resuming_state(state, 0)
```

**Rationale**: Python's type system doesn't automatically narrow types based on `hasattr` checks. The `cast` function provides an explicit way to tell the type checker the expected type after a runtime check.

**Impact**: Eliminates type errors for dynamic method calls while maintaining runtime safety.

**Files**: `src/nodetool/workflows/recovery.py`

**Date**: 2026-01-16
