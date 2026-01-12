### Python Environment Confusion

**Date Discovered**: 2024-01-10

**Context**: CI workflows sometimes fail due to incorrect Python environment assumptions

**Solution**:
- In GitHub CI: Use standard Python 3.11 with pip, dependencies pre-installed
- No conda activation needed in CI
- Use `uv sync --all-extras --dev` for installation

**Related Files**: `.github/workflows/*.yaml`

**Prevention**: Always check if running in CI environment before assuming conda
