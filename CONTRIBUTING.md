# Contributing to Nodetool Core

Thank you for considering contributing to Nodetool Core! This document provides guidelines and instructions for
contributing to the project.

## Development Process

1. Fork the repository
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
1. Make your changes
1. Run tests to ensure they pass (`make test`)
1. Commit your changes (`git commit -m 'Add some amazing feature'`)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

## Development Environment Setup

1. Clone your fork of the repository

1. Install development dependencies:

   ```bash
   # Using conda + uv (recommended)
   conda create -n nodetool python=3.11 pandoc ffmpeg -c conda-forge
   conda activate nodetool
   uv sync --all-extras --dev

   # Or using pip only
   pip install .
   pip install -r requirements-dev.txt
   ```

   This will install the package in development mode and install all development dependencies.

1. Set up environment configuration:

   ```bash
   cp .env.example .env.development.local
   # Edit .env.development.local with your API keys
   ```

   **Important:** Never commit `.env.*.local` files - they contain actual secrets and are gitignored.

   For a complete list of environment variables, see `.env.example` or the main README.md.

1. Install pre-commit hooks (optional):

   ```bash
   pre-commit install
   ```

## Code Style

This project uses:

- **Ruff** for linting and formatting (replaces Black + Flake8)
- **basedpyright** (via `ty` CLI) for type checking

You can run all style checks with:

```bash
make lint
```

Format your code with:

```bash
uv run ruff format .
```

## Testing

Write tests for all new features and bug fixes. Run the tests with:

```bash
make test
```

For verbose output:

```bash
make test-verbose
```

For test coverage report:

```bash
uv run pytest --cov=src
```

Run specific tests:

```bash
pytest tests/path/to/test_file.py
```

## Type Checking

Check for type errors with:

```bash
make typecheck
```

## Documentation

Update documentation for any changes to the public API. Comprehensive documentation is available at [docs.nodetool.ai](https://docs.nodetool.ai).

## Pull Request Process

1. Ensure your code passes all validation checks:
   ```bash
   make lint        # Check code style
   make test        # Run tests
   make typecheck   # Check types
   ```
1. Update documentation if needed (see [docs.nodetool.ai](https://docs.nodetool.ai))
1. Create a descriptive pull request with:
   - Clear description of changes
   - Links to related issues
   - Screenshots/logs for UI/CLI behavior changes
1. The PR should be reviewed by at least one maintainer
1. Once approved, a maintainer will merge your PR

## Commit Conventions

Follow **Conventional Commits** for your commit messages:
- `feat:` - New features
- `fix:` - Bug fixes
- `refactor:` - Code refactoring
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

Keep messages imperative and scoped.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and
welcoming community.
