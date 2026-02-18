# Anthropic Provider Return Type Mismatch

**Problem**: The `_setup_structured_output` method in `anthropic_provider.py` had a return type annotation that didn't match the actual return value.

**Solution**: Updated the return type from `tuple[list[dict] | None, dict | None, bool]` to `tuple[list[ToolParam] | list[dict] | None, dict | None, bool]` to match the actual return type from `format_tools()` which returns `list[ToolParam]`.

**Why**: The `format_tools()` method returns `list[ToolParam]` (a specific type from the anthropic library), but the return type annotation only allowed `list[dict] | None`. This caused a type checker error.

**Files**:
- `src/nodetool/providers/anthropic_provider.py:412`

**Date**: 2026-02-18
