# Type Check Warnings - January 2026

**Problem**: basedpyright reports 9 warnings in the codebase that don't cause failures but indicate potential issues.

**Warnings Summary**:
1. **Apple nodes module**: 7 warnings for dynamically created `ModuleType` attributes (e.g., `CreateCalendarEvent`, `ListCalendarEvents`, `SendMessage`, etc.) in `src/nodetool/nodes/apple/__init__.py`
2. **Ollama provider**: 1 warning for `invalid-method-override` - `convert_message` signature incompatible with parent `OpenAICompat.convert_message` (adds `use_tool_emulation: bool = False` parameter)

**Why**: The apple nodes use `ModuleType` to dynamically create node classes at runtime, which type checkers can't statically resolve. The Ollama provider's method override adds a new parameter that the parent class doesn't have.

**Files**:
- `src/nodetool/nodes/apple/__init__.py:71-114`
- `src/nodetool/providers/ollama_provider.py:254`

**Date**: 2026-01-18
