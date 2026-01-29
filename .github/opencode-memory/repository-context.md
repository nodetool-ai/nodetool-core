# NodeTool Core - Repository Context

This document provides essential context about the nodetool-core repository for AI coding agents.

## Project Overview

**NodeTool Core** is an open-source Python AI workflow engine that powers NodeTool. It provides a node-based DSL for building AI workflows, agent systems, and integrations with multiple AI providers.

## Key Technologies

- **Language**: Python 3.11+
- **Package Manager**: uv (preferred) or pip
- **Testing**: pytest
- **Linting**: ruff, black
- **Type Checking**: mypy, basedpyright
- **Build System**: Hatch (pyproject.toml)

## Project Structure

```
src/nodetool/
├── agents/         # LLM task planning and execution
├── api/            # FastAPI server with WebSocket support
├── chat/           # AI provider integrations
├── workflows/      # DAG-based workflow execution
├── storage/        # Multi-backend data persistence
├── models/         # Database adapters and schemas
├── dsl/            # Domain-specific language for workflows
├── metadata/       # Node metadata and types
└── ...

tests/              # Mirror structure of src/
docs/               # Documentation
examples/           # Code examples
```

## Development Environment

### Python Environment (CRITICAL)

**For GitHub CI / OpenCode Workflows:**
- Use standard Python 3.11 with pip
- Dependencies are pre-installed via `.github/workflows/copilot-setup-steps.yml`
- Run commands directly (no conda activation needed)

**For Local Development:**
- Use conda `nodetool` environment
- Or use `conda run -n nodetool <command>`

### Installation Commands

```bash
# Install using uv (preferred in CI)
uv sync --all-extras --dev

# Or using pip
pip install . && pip install -r requirements-dev.txt
```

## Validation Commands

These commands MUST pass before submitting PRs:

```bash
# Linting (uses ruff)
make lint
# or: ruff check .

# Type checking
make typecheck
# or: basedpyright

# Testing (quick)
make test
# or: pytest -q

# Testing (verbose)
make test-verbose
# or: pytest -v
```

## Coding Standards

### Style & Formatting
- **Black** style for formatting
- **Type hints** required for new/changed code
- **f-strings** preferred over format() or %
- **Async I/O** preferred where supported (many subsystems are async)

### Naming Conventions
- Files/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`

### Import Organization
- Standard library imports first
- Third-party imports second
- Local imports last
- Keep imports tidy and organized

## Testing Guidelines

- **Framework**: pytest
- **Location**: `tests/` mirrors `src/` structure
- **Naming**: 
  - Files: `test_*.py`
  - Functions: `test_*`
  - Classes: `Test*` (no `__init__`)
- **Environment**: Tests use `ENV=test` with in-memory storage
- **Coverage**: Include unit tests for logic, integration tests for I/O

## Architecture Patterns

- **Dependency Injection**: Components receive dependencies through constructors
- **Asynchronous Processing**: Heavy use of asyncio for non-blocking operations
- **Factory Pattern**: Provider factories create appropriate implementations
- **Strategy Pattern**: Different backends implement common interfaces
- **Observer Pattern**: WebSocket updates for real-time progress tracking

## Environment Variables

Key environment variables are documented in `.env.example`. Important categories:
- AI Provider APIs (OpenAI, Anthropic, Gemini, HuggingFace, etc.)
- Database & Storage (PostgreSQL, Supabase, S3)
- Vector DB (ChromaDB, Ollama)
- System tools (ffmpeg, ffprobe)

Use `.env.*.local` files for actual secrets (gitignored).

## Commit Conventions

Follow **Conventional Commits**:
- `feat:` - New features
- `fix:` - Bug fixes
- `refactor:` - Code refactoring
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

Keep messages imperative and scoped.

## Documentation

- Main docs: https://docs.nodetool.ai
- API changes require documentation updates
- Include screenshots/logs for UI/CLI behavior changes

## Common Pitfalls to Avoid

1. **Don't use system Python** - Use the provided environment setup
2. **Don't commit secrets** - Use `.env.*.local` files
3. **Don't break async patterns** - Avoid blocking calls in async code
4. **Don't skip validation** - Always run lint, test, and typecheck
5. **Don't make unrelated changes** - Keep PRs focused and minimal

## System Dependencies

The project requires these system-level tools:
- ffmpeg (media processing)
- libgl1, libglib2.0-0 (for certain Python packages)
- pandoc (documentation)

These are installed in CI via the setup steps.
