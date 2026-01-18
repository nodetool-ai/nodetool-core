# Convert Message Method Override Fix

**Problem**: The `convert_message` method in `ollama_provider.py` triggered an `invalid-method-override` type warning because it was a sync method overriding an async method from the `OpenAICompat` base class.

**Solution**: Added `# type: ignore[override]` comment to the method definition to explicitly acknowledge the intentional override pattern. This follows the same pattern used by other providers (`anthropic_provider.py`, `huggingface_provider.py`) that also have sync `convert_message` methods.

**Files**:
- `src/nodetool/providers/ollama_provider.py:254`

**Date**: 2026-01-18
