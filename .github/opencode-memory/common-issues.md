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

### Deprecated Typing Imports
**Date Discovered**: 2026-01-10
**Context**: Codebase uses deprecated `typing.Dict`, `typing.List`, `typing.Optional`, `typing.Type` imports which trigger ruff UP035/UP006/UP045 warnings
**Solution**: 
- Remove deprecated imports from typing module
- Use native Python 3.10+ syntax: `dict[...]` instead of `Dict[...]`, `list[...]` instead of `List[...]`, `X | None` instead of `Optional[X]`, `type` instead of `Type[X]`
- Run `ruff check --select=UP --fix` to auto-fix many issues
- Manually fix any remaining cases and verify with `ruff check`
**Related Files**: Multiple files in `src/nodetool/agents/`, `src/nodetool/api/`, `src/nodetool/providers/`
**Prevention**: Use modern Python type syntax in new code; run `ruff check` regularly to catch deprecation warnings

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
