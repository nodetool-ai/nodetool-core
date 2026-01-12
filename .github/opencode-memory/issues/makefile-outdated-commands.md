### Makefile Outdated Commands

**Date Disferenced**: 2026-01-10

**Context**: Makefile used `basedpyright` and `pytest -q` but CI uses `ty` and `pytest -n auto`

**Solution**: Updated Makefile to use:
  - `uv run ty check src` with appropriate ignore flags
  - `uv run pytest -n auto -q` for parallel test execution

**Related Files**: `Makefile`, `.github/workflows/test.yml`, `.github/workflows/typecheck.yml`

**Prevention**: Keep Makefile in sync with CI workflows
