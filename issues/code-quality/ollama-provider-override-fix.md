# OllamaProvider convert_message Liskov Override Fix

**Problem**: `OllamaProvider.convert_message` had an incompatible signature with parent `OpenAICompat.convert_message`:
- Parent: `async def convert_message(self, message: Message) -> Any`
- Child: `def convert_message(self, message: Message, use_tool_emulation: bool = False) -> Dict[str, Any]`

This violated the Liskov Substitution Principle, causing type checker warnings.

**Solution**: Updated `OllamaProvider` to properly override the parent method:
1. Made `convert_message` async to match parent signature
2. Made `_prepare_request_params` async (since it calls `convert_message`)
3. Updated call sites to use `await` for both methods

**Files**: `src/nodetool/providers/ollama_provider.py`

**Date**: 2026-01-18
