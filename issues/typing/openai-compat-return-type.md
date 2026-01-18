# OpenAICompat Return Type Fix

**Problem**: Base class `OpenAICompat.convert_message()` had inferred return type from return statements, causing type mismatch with Ollama provider override that returns `Dict[str, Any]`.

**Solution**: Added explicit return type `Any` to `OpenAICompat.convert_message()` method to allow overrides with different return types.

**Files**:
- `src/nodetool/providers/openai_compat.py`

**Date**: 2026-01-18
