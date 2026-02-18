# Actor Type Checking Import

**Problem**: The `MessageEnvelope` import in `actor.py` was not in a type-checking block, causing a TC001 linting error.

**Solution**: Moved `from nodetool.workflows.inbox import MessageEnvelope` into the existing `TYPE_CHECKING` block.

**Why**: `MessageEnvelope` is only used in type annotations (for `deferred_control` parameter lists) and not at runtime. Moving it to the type-checking block improves import efficiency and follows best practices for type-only imports.

**Files**:
- `src/nodetool/workflows/actor.py:51`

**Date**: 2026-02-18
