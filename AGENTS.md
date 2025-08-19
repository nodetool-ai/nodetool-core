# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/nodetool/` (e.g., `agents/`, `api/`, `chat/`, `common/`, `dsl/`, `workflows/`).
- Tests: `tests/` mirrors the source layout (e.g., `tests/agents`, `tests/api`).
- Docs and examples: `docs/`, `examples/`.
- Packaging: Poetry project (`pyproject.toml`), console entry `nodetool`.

## Build, Test, and Development Commands
- `conda activate nodetool`
- for tests `pytest tests`

## Coding Style & Naming Conventions
- Language: Python 3.11, type hints required for new/changed code.
- Formatting: Black style; keep imports and whitespace tidy; prefer f‑strings.
- Linting: Ruff for quick rules; Flake8/Mypy/Pylint configs exist for CI parity.
- Names: `snake_case` for files/functions, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- Modules: keep public APIs under `src/nodetool/...` with small, focused modules.

## Testing Guidelines
- Framework: pytest; locate tests under `tests/` with structure mirroring `src/`.
- Naming: files `test_*.py`, functions `test_*`, classes `Test*` (no `__init__`).
- Running: `poetry run pytest -q` for quick checks; add fixtures in `tests/conftest.py`.
- Scope: include unit tests for logic and lightweight integration tests for I/O and async paths.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (`feat:`, `fix:`, `refactor:`, etc.); keep messages imperative and scoped.
- PRs: include a clear description, linked issues, and screenshots/logs if UI/CLI behavior changes.
- Checks: ensure `pytest`, `ruff`, `black`, and `mypy` pass locally; update docs/examples when APIs change.

## Security & Configuration Tips
- Do not commit secrets; use `.env` locally (git‑ignored). Key envs: `FFMPEG_PATH`, `FFPROBE_PATH` when handling media.
- Prefer async I/O where supported (many subsystems are async) and avoid blocking calls in hot paths.
