# Type Fixes for Agent Modules

**Problem**: Multiple type errors in agent modules (graph_planner.py, task_planner.py, step_executor.py, task_executor.py) including:
- Missing type arguments for `dict` generic class
- Incompatible async generator return types
- Class variable overriding instance variable
- Missing type arguments for `DiGraph` generic class

**Solution**: Fixed type annotations across multiple files:

1. **graph_planner.py**:
   - Added type ignore for `ClassVar` override issue
   - Updated `_run_workflow_design_phase` return type to allow `None` in tuple
   - Updated `create_graph` return type to include `ToolCall`
   - Changed `_build_nodes_and_edges_from_specifications` to return `list[Node]` and `list[Edge]` instead of `list[dict]`
   - Imported `Node` and `Edge` from `api_graph` module

2. **task_planner.py**:
   - Added `ClassVar` to base `Tool` class in `base.py` to fix override conflict
   - Added type arguments to all `dict` usages (e.g., `dict[str, Any]`)
   - Added type ignore comment for `DiGraph` return type

3. **step_executor.py**:
   - Added type arguments to return type `dict | str` -> `dict[str, Any] | str`

4. **task_executor.py**:
   - Added `StepResult` to return type of `execute_tasks`
   - Imported `StepResult` and `Chunk` from `workflows.types`
   - Fixed duplicate imports

**Why**: These fixes resolve type checking errors and improve type safety across the agent modules.

**Files**:
- `src/nodetool/agents/graph_planner.py`
- `src/nodetool/agents/task_planner.py`
- `src/nodetool/agents/step_executor.py`
- `src/nodetool/agents/task_executor.py`
- `src/nodetool/agents/tools/base.py`

**Date**: 2026-01-18
