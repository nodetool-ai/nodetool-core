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

### Async File I/O in Async Functions
**Date Discovered**: 2026-01-12
**Context**: Async functions using blocking `open()` calls trigger ASYNC230 lint errors and can block the event loop
**Solution**: 
- For subprocess file handles that require synchronous IO: Use `loop.run_in_executor()` with a sync helper function
- For regular file reads: Use `aiofiles` library for async file operations
**Related Files**: `src/nodetool/agents/agent_evaluator.py`
**Prevention**: Use `aiofiles` for async file operations; use `asyncio.to_thread()` or `run_in_executor()` for blocking file I/O that must remain sync
### Overly Broad Exception Handling
**Date Discovered**: 2026-01-10
**Context**: Multiple files use `except Exception:` or `except Exception as exc:` which catches all exceptions including system-exiting ones (KeyboardInterrupt, SystemExit) and makes debugging difficult.
**Solution**: Replace `except Exception:` with specific exception types:
- `except (TypeError, ValueError):` for JSON serialization errors
- `except (KeyError, ValueError, base64.binascii.Error):` for data parsing errors
- `except requests.RequestException:` for HTTP request errors
- `except (OSError, ValueError):` for file I/O and audio conversion errors
**Related Files**:
- `src/nodetool/providers/comfy_runpod_provider.py` (lines 118, 122)
- `src/nodetool/providers/openai_compat.py` (lines 72, 108)
- `src/nodetool/providers/llama_provider.py` (lines 162, 276, 579, 618)
**Prevention**: Use ruff rule `TRY302` (raise from) and `TRY201` (bare raise) when appropriate, and be specific about exception types. Log exceptions instead of silently swallowing them.
### Print Statement Usage Instead of Proper Logging
**Date Discovered**: 2026-01-11
**Context**: Multiple files were using `print()` for error handling and status messages instead of proper Python logging
**Solution**: Replace print statements with appropriate logger calls:
- Add `import logging` and create module logger: `logger = logging.getLogger("nodetool.module_name")`
- Replace `print(f"Error: {e}")` with `logger.error("Error: %s", e)`
- Replace `print(f"Warning: {msg}")` with `logger.warning("%s", msg)`
- Replace `print(f"Info: {msg}")` with `logger.info("%s", msg)`
- Use `logger.exception()` for exceptions (automatically includes traceback)
- Remove unused `traceback` imports when switching to `logger.exception()`
**Related Files**:
- `src/nodetool/providers/comfy_api.py` (~60 print statements converted)
- `src/nodetool/deploy/runpod_api.py` (critical error logging fixed)
- `src/nodetool/deploy/admin_routes.py` (5 print statements converted)
- `src/nodetool/deploy/workflow_routes.py` (2 print statements converted)
**Prevention**: Add linting rule (e.g., flake8-print) to detect print statements, or use mypy to discourage print in production code

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
