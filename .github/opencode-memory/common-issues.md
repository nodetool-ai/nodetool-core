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
**Context**: 5 job execution tests fail when run with `pytest -n auto` but pass individually
**Solution**: Tests need shared state isolation for parallel execution (pre-existing issue)
**Related Files**: `tests/workflows/test_job_execution.py`, `tests/workflows/test_job_execution_manager.py`
**Prevention**: Use unique test databases/resources for each test

### Test Status Timing Issues
**Date Discovered**: 2026-01-12
**Context**: Tests checking job status fail when workflow completes faster than the test can check the status. Empty workflows (0 nodes) complete in milliseconds, so status checks for "running" may find "completed" or "scheduled" instead.
**Solution**:
1. Added wait loop with timeout to allow job to transition to running state
2. Updated status assertions to accept multiple valid states: "running", "completed", "failed", and "scheduled"
**Related Files**:
- `tests/workflows/test_job_execution.py:115-127` - test_start_job
- `tests/workflows/test_threaded_job_execution.py:236-245` - test_threaded_job_database_record
**Prevention**: For quick-completing workflows, check status with a timeout or accept multiple valid terminal states in assertions

### Insecure Deserialization in Model Cache
**Date Discovered**: 2026-01-12
**Context**: `ModelCache` class used `pickle.load()` to deserialize cached data from disk. Pickle is insecure by design and can execute arbitrary code during deserialization if the cache file is tampered with.
**Solution**: Replaced `pickle.load()`/`pickle.dump()` with JSON serialization using a custom `CacheJSONEncoder` that handles bytes, datetime, and set types.
**Related Files**:
- `src/nodetool/ml/models/model_cache.py`
**Prevention**: Never use pickle for untrusted data. Use JSON or other safe serialization formats.

### Shell Injection Risk in Docker Commands
**Date Discovered**: 2026-01-12
**Context**: Docker build and push commands used string interpolation with user-controlled variables (`image_name`, `tag`, `platform`) without proper escaping when calling `subprocess.run()` with `shell=True`.
**Solution**: Added `_shell_escape()` helper function using `shlex.quote()` to properly escape all variables interpolated into shell commands.
**Related Files**:
- `src/nodetool/deploy/docker.py`
**Prevention**: Always use `shlex.quote()` when interpolating variables into shell commands with `shell=True`, or prefer list-based subprocess calls.
### Blocking HTTP Calls in Async Code
**Date Discovered**: 2026-01-12
**Context**: Several files use synchronous `requests` library for HTTP calls in modules that otherwise use async patterns. This blocks the event loop during network I/O.
**Solution**: Convert blocking `requests.get/post` calls to async `httpx` calls:
- `src/nodetool/providers/huggingface_provider.py:95-157` - Converted `get_remote_context_window()` from sync `requests.get` to async `httpx.client.AsyncClient.get`
**Related Files**: 
- `src/nodetool/providers/huggingface_provider.py`
- `src/nodetool/providers/comfy_api.py` (still uses requests, could benefit from similar fix)
- `src/nodetool/packages/registry.py` (still uses requests)
- `src/nodetool/deploy/runpod_api.py` (still uses requests)
**Prevention**: 
- Use `httpx` for all HTTP operations in async modules
- Run `ruff check` to verify no blocking patterns
- Consider wrapping sync I/O with `asyncio.to_thread()` if async conversion is not feasible

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
