# Keeping Makefile and Documentation in Sync

**Insight**: When updating Makefile targets, update all documentation files in the same commit

**Rationale**: Multiple documentation files (README.md, CONTRIBUTING.md, AGENTS.md, CLAUDE.md) may reference the same commands. Keeping them synchronized prevents confusion.

**Example**: When changing from Flake8/Black to Ruff:
1. Update Makefile
2. Search for all references to old tools:
   ```bash
   grep -r "flake8\|black\|mypy" *.md docs/
   ```
3. Update all found references in the same commit

**Impact**: Prevents the kind of drift seen where CONTRIBUTING.md referenced `make dev-install` and `make format` targets that never existed in the Makefile

**Files**: Makefile, CONTRIBUTING.md, README.md, AGENTS.md, CLAUDE.md

**Date**: 2026-02-16
