# Missing Type Argument for Generic Dict - 2026-01-14

**Problem**: The `output_schema` parameter in `Agent.__init__` used `dict | None` without type arguments, causing type checker errors about missing type arguments for generic class `dict`.

**Solution**: Changed `output_schema: dict | None = None` to `output_schema: dict[str, Any] | None = None` in `src/nodetool/agents/agent.py:508`.

**Files**:
- `src/nodetool/agents/agent.py`

**Date**: 2026-01-14
