# Documentation Command Validation

**Insight**: Always verify documentation commands against actual Makefile and CI workflow files before recommending them

**Rationale**: Outdated commands in documentation (like `make dev-install`, `make format`, `make test-cov`) that don't exist in the Makefile cause contributor frustration and wasted time

**Example**: Check Makefile for available targets before documenting them:
```bash
# List all PHONY targets in Makefile
grep "^.PHONY:" Makefile
grep "^[a-z-]*:" Makefile | grep -v "^\s*#"

# Or just check what targets exist
make -n target_name  # Returns error if target doesn't exist
```

**Impact**: Reduces contributor onboarding friction and support burden

**Files**: CONTRIBUTING.md, Makefile

**Date**: 2026-02-16
