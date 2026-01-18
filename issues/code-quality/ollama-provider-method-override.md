# OllamaProvider Method Override Fix

**Problem**: The `convert_message` method in `OllamaProvider` violated the Liskov Substitution Principle by having a different signature than the parent `OpenAICompat` class:
- Parent: `async def convert_message(self, message: Message) -> Any`
- Ollama: `def convert_message(self, message: Message, use_tool_emulation: bool = False) -> Dict[str, Any]`

Issues:
1. Parent method was async, Ollama's was sync
2. Different return types (`Any` vs `Dict[str, Any]`)
3. Extra `use_tool_emulation` parameter

**Solution**: Made the Ollama method properly async to match the parent class signature, and updated internal callers to use `await`:
- Changed `convert_message` from `def` to `async def`
- Changed `_prepare_request_params` from `def` to `async def`
- Updated callers to use `await self.convert_message(...)` and `await self._prepare_request_params(...)`

**Files Modified**:
- `src/nodetool/providers/ollama_provider.py`

**Date**: 2026-01-18
