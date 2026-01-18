# Type Cast Quoting Fix

**Problem**: Type expressions in `typing.cast()` calls lacked quotes, causing TC006 linting violations.

**Solution**: Added quotes to type expressions in cast calls across multiple files:
- `src/nodetool/integrations/vectorstores/chroma/provider_embedding_function.py:112`
- `src/nodetool/providers/openai_provider.py:2202-2205`

**Why**: The project's ruff configuration requires quoted type expressions in cast() calls for better compatibility.

**Files**:
- `src/nodetool/integrations/vectorstores/chroma/provider_embedding_function.py`
- `src/nodetool/providers/openai_provider.py`

**Date**: 2026-01-18
