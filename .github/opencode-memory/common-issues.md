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
