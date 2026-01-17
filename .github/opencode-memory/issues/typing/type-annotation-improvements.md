# Type Annotation Improvements

**Problem**: Multiple files had missing type arguments for generic types like `dict` and `list`, causing type checker errors.

**Solution**: Added explicit type arguments to generic type annotations:
- `dict` → `dict[str, Any]`
- `dict` → `list[dict[str, Any]]`
- `list` → `list[Any]`

**Files Fixed**:
- `src/nodetool/agents/docker_runner.py:13` - `_run(cfg: dict)` → `cfg: dict[str, Any]`
- `src/nodetool/agents/graph_planner.py:737, 750` - `list` type annotations
- `src/nodetool/agents/step_executor.py:555, 1494` - `context: dict` → `context: dict[str, Any]`, `tool_args: dict` → `tool_args: dict[str, Any]`
- `src/nodetool/agents/serp_providers/data_for_seo_provider.py:73` - `payload: list[dict]` → `payload: list[dict[str, Any]]`

**Additional Fixes**:
- `src/nodetool/agents/agent.py:626` - Added assertion and cast for `workspace_dir` to handle `str | None` type
- `src/nodetool/agents/agent.py:708` - Added defensive check for `item.step` before accessing `.id`
- `src/nodetool/agents/step_executor.py:717` - Added None check for `result_schema` before validation
- `src/nodetool/agents/step_executor.py:858` - Added `LogUpdate` to async generator return type

**Date**: 2026-01-15
