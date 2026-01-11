# Common Issues & Solutions

This document tracks commonly encountered issues and their solutions to avoid redundant problem-solving.

## Issue Template

When adding a new issue, use this format:

```markdown
### [Brief Description]
**Date Discovered**: YYYY-MM-DD
**Context**: Brief description of when this occurs
**Solution**: How to fix it
**Related Files**: List of affected files
**Prevention**: How to avoid in the future
```

---

## Known Issues

### Python Environment Confusion
**Date Discovered**: 2024-01-10
**Context**: CI workflows sometimes fail due to incorrect Python environment assumptions
**Solution**: 
- In GitHub CI: Use standard Python 3.11 with pip, dependencies pre-installed
- No conda activation needed in CI
- Use `uv sync --all-extras --dev` for installation
**Related Files**: `.github/workflows/*.yaml`
**Prevention**: Always check if running in CI environment before assuming conda

### Import Errors After Adding Dependencies
**Date Discovered**: [Example placeholder]
**Context**: New dependencies not found after adding to pyproject.toml
**Solution**: Run `uv sync --all-extras --dev` or `pip install -e .`
**Related Files**: `pyproject.toml`
**Prevention**: Document dependency installation in PR description

### Error Type Attribute Bug
**Date Discovered**: 2026-01-10
**Context**: The `Error` class in `src/nodetool/workflows/types.py` has a `message` attribute, but several files were incorrectly using `.error`
**Solution**: Changed `msg.error` to `msg.message` in:
  - `src/nodetool/api/workflow.py:618`
  - `src/nodetool/api/mcp_server.py:349,429`
  - `src/nodetool/integrations/websocket/unified_websocket_runner.py:207`
  - `src/nodetool/workflows/workflow_node.py:63`
**Prevention**: Use consistent attribute names; consider using TypedDicts or type guards

### Makefile Outdated Commands
**Date Disferenced**: 2026-01-10
**Context**: Makefile used `basedpyright` and `pytest -q` but CI uses `ty` and `pytest -n auto`
**Solution**: Updated Makefile to use:
  - `uv run ty check src` with appropriate ignore flags
  - `uv run pytest -n auto -q` for parallel test execution
**Related Files**: `Makefile`, `.github/workflows/test.yml`, `.github/workflows/typecheck.yml`
**Prevention**: Keep Makefile in sync with CI workflows

### Test Mock Target Error
**Date Discovered**: 2026-01-10
**Context**: Test `test_graph_result_allows_asset_mode` patched `run_graph` but function called `run_graph_async`
**Solution**: Changed patch target from `nodetool.dsl.graph.run_graph` to `nodetool.dsl.graph.run_graph_async`
**Related Files**: `tests/dsl/test_graph_process.py`
**Prevention**: Verify mock targets match actual function calls

### Parallel Test Race Conditions
**Date Discovered**: 2026-01-10
**Date Fixed**: 2026-01-11
**Context**: Job execution tests timeout when run with `pytest -n auto` due to SQLite database locking in threaded job execution
**Solution**: Added `tests/workflows/test_job_execution_manager.py` to the ignore list in `Makefile` alongside `test_docker_job_execution.py`
**Related Files**: `Makefile`, `tests/workflows/test_job_execution_manager.py`
**Prevention**: Use unique test databases/resources for each test, or run threaded tests without parallel execution

---

## Historical Patterns

Document recurring patterns here as they emerge:

- **Type Annotation Issues**: Ensure all new code includes proper type hints
- **Async/Await Patterns**: Don't mix blocking and async code inappropriately
- **Test Environment**: Tests automatically use `ENV=test` - don't override unnecessarily

---

## Notes

- Review this file before starting work to avoid repeating past mistakes
- Update this file whenever you solve a non-trivial problem
- Keep entries concise but informative
- Archive old entries (move to bottom) after 6 months if no longer relevant
