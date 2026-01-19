# Ollama convert_message Method Override Warning

**Problem**: `OllamaProvider.convert_message()` has signature `convert_message(self, message: Message, use_tool_emulation: bool = False) -> Dict[str, Any]` which is incompatible with parent class `OpenAICompat.convert_message(self, message: Message) -> Any`. The extra `use_tool_emulation` parameter violates the Liskov Substitution Principle.

**Solution**: Either:
1. Remove the `use_tool_emulation` parameter from Ollama's override
2. Add the parameter to the base class with a default value
3. Use a different method name for Ollama-specific functionality

**Files**:
- `src/nodetool/providers/ollama_provider.py:254`
- `src/nodetool/providers/openai_compat.py:148`

**Date**: 2026-01-19
