# CONTRIBUTING.md Outdated Commands

**Problem**: CONTRIBUTING.md documented nonexistent Makefile targets (`make dev-install`, `make format`, `make test-cov`, `make docs`) and outdated tooling (Flake8, MyPy instead of Ruff)

**Solution**: Updated CONTRIBUTING.md to:
- Use `uv sync --all-extras --dev` for installation (no `make dev-install` target exists)
- Use `uv run ruff format .` for formatting (no `make format` target exists)
- Use `uv run pytest --cov=src` for coverage (no `make test-cov` target exists)
- Reference Ruff for linting (replaces Flake8)
- Reference basedpyright/ty for type checking (replaces MyPy)
- Link to docs.nodetool.ai instead of local `make docs` target

**Why**: The Makefile only has `lint`, `typecheck`, `test`, and `test-verbose` targets. Old documentation confused contributors and caused frustration when commands didn't work.

**Files**: CONTRIBUTING.md

**Date**: 2026-02-16
