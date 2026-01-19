# Ollama Provider convert_message Method Override

**Problem**: The `convert_message` method in `OllamaProvider` violated the Liskov Substitution Principle by having a different signature than the parent class `OpenAICompat.convert_message`. Specifically:
- Parent: `async def convert_message(self, message: Message) -> Any`
- Child: `def convert_message(self, message: Message, use_tool_emulation: bool = False) -> Dict[str, Any]`

**Solution**: Made the child method properly async and updated downstream calls:
1. Changed `convert_message` to `async def convert_message`
2. Made `_prepare_request_params` async to properly await the conversion
3. Updated callers to use `await` with `asyncio.gather` for parallel message conversion

**Files**: `src/nodetool/providers/ollama_provider.py`

**Date**: 2026-01-18
