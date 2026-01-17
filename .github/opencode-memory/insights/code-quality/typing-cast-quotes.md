# Ruff TC006: typing.cast() Type Expressions

**Insight**: When using `typing.cast()`, the type parameter should be a quoted string type expression (e.g., `cast("Literal['tool']", value)`) rather than the raw type (`cast(Literal["tool"], value)`).

**Rationale**: This satisfies type checker requirements and avoids ambiguity. The quoted string form is explicitly supported by Python's typing system for forward references and type expressions.

**Example**:
```python
# Before (TC006 violation):
role=cast(Literal["tool"], message.role)

# After (compliant):
role=cast("Literal['tool']", message.role)
```

**Impact**: Eliminates 21 TC006 lint violations across provider files.

**Files**: src/nodetool/providers/anthropic_provider.py, src/nodetool/providers/openai_compat.py, src/nodetool/providers/openai_provider.py, src/nodetool/io/media_fetch.py

**Date**: 2026-01-17
