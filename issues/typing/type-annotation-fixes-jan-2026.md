# Type Annotation Fixes - January 2026

**Problem**: Multiple functions in `src/nodetool/agents/agent.py` had missing or incorrect type annotations:
- `_get_cpu_limit()` used `Any` for `resource_limits` parameter
- `_wrap_command_with_cpu_limit()` used `Any` for `resource_limits` parameter
- `AgentRunner.__init__()` missing return type annotation
- `Agent.__init__()` missing return type annotation and used `Any` for `resource_limits`
- `wrap_subprocess_command()` had incorrect return type `tuple[list[str], Any]`
- `cleanup_subprocess_wrapper()` used `Any` for `cleanup_data` parameter
- `test_docker_feature()` and `test_sandbox_feature()` missing return type annotations

**Solution**: Fixed type annotations across the agent module:
1. Imported `ResourceLimits` from `nodetool.workflows.run_job_request`
2. Changed `resource_limits: Any | None` to `resource_limits: ResourceLimits | None`
3. Changed `cleanup_data: Any` to `cleanup_data: str | None`
4. Changed return type from `tuple[list[str], Any]` to `tuple[list[str], str | None]`
5. Added `-> None` return type annotations to `__init__` methods and test functions

**Why**: Proper type annotations improve code quality, enable better IDE support, and catch bugs at type-checking time. The ruff linter was enforcing these rules via ANN201, ANN204, and ANN401.

**Files Modified**:
- `src/nodetool/agents/agent.py`
- `src/nodetool/workflows/io.py` (import ordering fix)

**Date**: 2026-01-22
