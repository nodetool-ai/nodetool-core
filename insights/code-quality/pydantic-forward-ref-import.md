# Pydantic Forward Reference Import Requirement

**Insight**: An unused import may actually be required for Pydantic's schema generation when dealing with third-party types that have forward references.

**Example**: In `src/nodetool/types/model.py`, the import:
```python
from huggingface_hub.inference._providers import PROVIDER_T
```

Appears unused (no direct reference in the code), but is required because:
1. `ModelInfo` from `huggingface_hub` has type annotations that reference `PROVIDER_T`
2. `CachedFileInfo.model_rebuild()` triggers Pydantic schema generation
3. Pydantic evaluates type annotations in the module's namespace
4. Without `PROVIDER_T` in the namespace, schema generation fails with `NameError`

**Rationale**: Third-party libraries may use forward references that leak into their public API type annotations. When Pydantic generates schemas, it needs those references available in the namespace.

**What to Do**: If removing an "unused" import causes Pydantic schema errors:
1. Check if the imported type is referenced in any field type annotations
2. Consider whether the import is needed for forward reference resolution
3. Document why the import is required if kept

**Files**: `src/nodetool/types/model.py`

**Date**: 2026-01-22
