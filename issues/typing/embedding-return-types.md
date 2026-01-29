# Embedding Return Type Fixes

**Problem**: Multiple type mismatches in embedding function return types:
- `provider_embedding_function.py`: Return type `Embeddings` didn't match actual return type
- `ollama_provider.py`: Return type `list[list[float]]` didn't match `Sequence[Sequence[int | float]]`
- `openai_provider.py`: Config dict values had union types that didn't match expected types

**Solution**: Added `cast()` calls to properly type the return values:
1. `provider_embedding_function.py`: Cast embeddings to `Embeddings`
2. `ollama_provider.py`: Cast embeddings to `list[list[float]]`
3. `openai_provider.py`: Cast config values to `str`, `str`, and `int`

**Files**:
- `src/nodetool/integrations/vectorstores/chroma/provider_embedding_function.py`
- `src/nodetool/providers/ollama_provider.py`
- `src/nodetool/providers/openai_provider.py`

**Date**: 2026-01-18
