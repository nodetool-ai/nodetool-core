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

### Ruff Import Sorting in Test Files
**Date Discovered**: 2026-01-10
**Context**: Imports inside test functions need to follow isort order: standard library → third-party → local
**Solution**: Use `ruff check --fix` to auto-fix, or manually order imports correctly
**Related Files**: `tests/chat/providers/test_gemini_provider.py`
**Prevention**: Run `ruff check` before committing, use IDE integration for real-time feedback

### graph_result Function Calling Pattern
**Date Discovered**: 2026-01-10
**Context**: `graph_result` called `run_graph_async` directly, bypassing `run_graph` wrapper that tests mock
**Solution**: Changed `graph_result` to call `run_graph` instead, which handles sync/async properly
**Related Files**: `src/nodetool/dsl/graph.py`, `tests/dsl/test_graph_process.py`
**Prevention**: When creating convenience wrappers, ensure they use the same entry points that tests mock

### Async Test Fixtures for Sync Functions
**Date Discovered**: 2026-01-10
**Context**: Test fixture was async but patched function (`run_graph`) is sync, causing coroutine objects to be returned
**Solution**: Remove `async` from fake function definition since `run_graph` is not async
**Related Files**: `tests/dsl/test_graph_process.py`
**Prevention**: Match the async/sync nature of the function being mocked in test fixtures

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
