### Import Errors After Adding Dependencies

**Date Discovered**: [Example placeholder]

**Context**: New dependencies not found after adding to pyproject.toml

**Solution**: Run `uv sync --all-extras --dev` or `pip install -e .`

**Related Files**: `pyproject.toml`

**Prevention**: Document dependency installation in PR description
