# Build, Test, Lint

## Required Commands

```bash
make typecheck
make lint
make test
```

## Notes

- Use Python 3.11+ with dependencies installed via `uv sync --all-extras --dev`.
- Keep async code non-blocking; prefer `async`/`await` patterns.
