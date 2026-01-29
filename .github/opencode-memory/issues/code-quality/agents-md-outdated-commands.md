### AGENTS.md Outdated Commands

**Date Discovered**: 2026-01-12

**Context**: AGENTS.md documented outdated commands (`black .`, `mypy .`) that don't match the Makefile or CI workflows

**Solution**: Updated AGENTS.md to use:
- `uv sync --all-extras --dev` for installation
- `make lint` / `uv run ruff check .` for linting
- `make typecheck` / `uv run ty check src` for type checking
- `make test` / `uv run pytest -n auto -q` for tests

**Related Files**: `AGENTS.md`

**Prevention**: Keep AGENTS.md in sync with Makefile and CI workflows
