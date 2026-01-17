# Pydantic Config Deprecation and Unused Imports Cleanup

**Problem**: The codebase had deprecated Pydantic v1 patterns (`class Config:`) and unused imports in source and example files that were causing lint warnings and deprecation warnings.

**Solution**: 
- Replaced deprecated `class Config: from_attributes = True` with `model_config = ConfigDict(from_attributes=True)` in:
  - `src/nodetool/api/cost.py:33`
  - `src/nodetool/api/job.py:42`
- Removed unused imports from source files:
  - `src/nodetool/agents/agent.py`: Removed unused `Step` import
  - `src/nodetool/api/job.py`: Removed unused `ConditionBuilder` import
  - `src/nodetool/integrations/websocket/unified_websocket_runner.py`: Removed unused `get_or_create_tracer` import
  - `src/nodetool/api/asset.py`: Removed unused `Dict`, `List`, `Tuple` from typing, replaced with modern `dict`, `list`, `tuple`
- Removed unused imports from example and script files:
  - `examples/chromadb_research_agent.py`: Removed unused `ChromaMarkdownSplitAndIndexTool` and `ConvertPDFToMarkdownTool`
  - `examples/graph_planner_integration.py`: Removed unused `OpenAIProvider`
  - `examples/test_google_agent.py`: Removed unused `PlanningUpdate`
  - `examples/test_simple_agent.py`: Removed unused `GoogleSearchTool`
  - `scripts/test_default_nodes.py`: Removed unused `json` and `sys` imports, fixed `Dict[str, Any]` to `dict[str, Any]`
- Fixed deprecated `typing.List` usage in `examples/learning_path_generator.py`

**Why**: Pydantic v2 deprecates class-based Config in favor of `model_config = ConfigDict(...)`. Using modern patterns ensures forward compatibility and eliminates deprecation warnings. Removing unused imports improves code cleanliness and reduces confusion.

**Files**:
- `src/nodetool/api/cost.py`
- `src/nodetool/api/job.py`
- `src/nodetool/agents/agent.py`
- `src/nodetool/integrations/websocket/unified_websocket_runner.py`
- `src/nodetool/api/asset.py`
- `examples/chromadb_research_agent.py`
- `examples/graph_planner_integration.py`
- `examples/test_google_agent.py`
- `examples/test_simple_agent.py`
- `examples/learning_path_generator.py`
- `scripts/test_default_nodes.py`

**Date**: 2026-01-17
