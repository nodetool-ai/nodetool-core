# Important Insights & Learnings

This document captures valuable insights about the codebase that should be remembered for future work.

## Insight Template

When adding a new insight, use this format:

```markdown
### [Insight Title]
**Date**: YYYY-MM-DD
**Category**: Architecture / Testing / Performance / Security / etc.
**Insight**: The key learning or observation
**Impact**: How this affects development work
**Examples**: File paths or code references
```

---

## Architecture Insights

### Async-First Design
**Date**: 2024-01-10
**Category**: Architecture
**Insight**: The codebase is heavily async-oriented. Most subsystems (workflows, agents, storage, API) use asyncio extensively.
**Impact**: New code should prefer async/await patterns. Avoid blocking calls in hot paths.
**Examples**: `src/nodetool/workflows/`, `src/nodetool/agents/`, `src/nodetool/api/`

### Dependency Injection Pattern
**Date**: 2024-01-10
**Category**: Architecture
**Insight**: Components receive dependencies through constructors rather than creating them internally.
**Impact**: When adding new components, follow DI pattern. Makes testing easier and code more modular.
**Examples**: Throughout `src/nodetool/agents/`, `src/nodetool/workflows/`

---

## Testing Insights

### Test Environment Auto-Configuration
**Date**: 2024-01-10
**Category**: Testing
**Insight**: Tests automatically use `ENV=test` with in-memory storage and `/tmp/nodetool_test.db`.
**Impact**: Don't override test environment manually unless absolutely necessary. The auto-configuration ensures isolation.
**Examples**: `tests/conftest.py`

### Mirror Directory Structure
**Date**: 2024-01-10
**Category**: Testing
**Insight**: Test directory structure mirrors `src/` structure exactly.
**Impact**: When adding new source files, create corresponding test files in mirrored location.
**Examples**: `tests/agents/` mirrors `src/nodetool/agents/`

### Timing-Sensitive Test Design
**Date**: 2026-01-12
**Category**: Testing
**Insight**: Empty workflows (0 nodes) complete in milliseconds. Tests checking for "running" status may find "completed" or "scheduled" instead.
**Impact**: Use wait loops with timeouts when checking for status transitions. Accept multiple valid states in assertions for quick-completing operations.
**Examples**: `tests/workflows/test_job_execution.py`, `tests/workflows/test_threaded_job_execution.py`

---

## Code Quality Insights

### Validation Commands Must Pass
**Date**: 2024-01-10
**Category**: Code Quality
**Insight**: Three commands must pass before PR submission: `make lint`, `make test`, `make typecheck`
**Impact**: Always run all three before finalizing work. Don't skip any.
**Examples**: `Makefile`, `.github/workflows/`

### Conventional Commits Required
**Date**: 2024-01-10
**Category**: Code Quality
**Insight**: Commits follow Conventional Commits format (feat:, fix:, refactor:, etc.)
**Impact**: All commit messages must be properly formatted. Helps with changelog generation and history clarity.
**Examples**: Git commit history

---

## Security Insights

### Environment Variable Layering
**Date**: 2024-01-10
**Category**: Security
**Insight**: Configuration uses layered approach: defaults → `.env` → environment-specific → `.env.*.local` → env vars → YAML
**Impact**: Never commit `.env.*.local` files. Use them for actual API keys and secrets.
**Examples**: `.env.example`, `.env.development`, `.env.test`

### Avoid pickle for Disk Serialization
**Date**: 2026-01-12
**Category**: Security
**Insight**: Python's `pickle` module is insecure by design - it can execute arbitrary code during deserialization. Never use it for data that could be tampered with (e.g., cache files, user-uploaded data).
**Impact**: Replace pickle with JSON for disk serialization. Use custom encoders for special types (bytes, datetime, sets).
**Examples**: `src/nodetool/ml/models/model_cache.py`

### Shell Command Escaping
**Date**: 2026-01-12
**Category**: Security
**Insight**: When using `subprocess.run()` with `shell=True`, any variables interpolated into the command string must be properly escaped to prevent shell injection. Even "trusted" values should be escaped.
**Impact**: Use `shlex.quote()` for all variable interpolation, or prefer list-based subprocess calls with `shell=False`.
**Examples**: `src/nodetool/deploy/docker.py`

### ast.literal_eval is Safe
**Date**: 2026-01-12
**Category**: Security
**Insight**: Unlike `eval()`, `ast.literal_eval()` only evaluates literal Python expressions (strings, numbers, tuples, lists, dicts, booleans, None). It cannot execute arbitrary code.
**Impact**: Use `ast.literal_eval()` as a safe fallback for parsing JSON with single quotes or minor syntax issues.
**Examples**: `src/nodetool/utils/message_parsing.py`

---

## Performance Insights

### Streaming-First Execution
**Date**: 2024-01-10
**Category**: Performance
**Insight**: Workflow engine uses actor-based execution with one actor per node. Nodes handle data piece-by-piece (streaming) where possible.
**Impact**: Design new nodes to support streaming when dealing with large datasets or long-running operations.
**Examples**: `src/nodetool/workflows/`

---

## Notes

- Review this file when starting new features or debugging complex issues
- Add insights that would save significant time if known earlier
- Focus on non-obvious patterns and decisions
- Keep insights actionable and specific
