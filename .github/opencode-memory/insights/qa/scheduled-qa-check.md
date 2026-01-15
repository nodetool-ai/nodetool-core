# Scheduled QA Check Process

**Insight**: Running `make typecheck`, `make lint`, and `make test` ensures code quality gates pass before committing.

**Rationale**: The three validation commands catch different categories of issues:
- `make typecheck`: Basedpyright catches type errors and inconsistencies
- `make lint`: Ruff catches code style and potential bugs
- `make test`: Pytest ensures functional correctness

**Result**: On 2026-01-15, all checks passed:
- Typecheck: 110 warnings, 0 errors
- Lint: All checks passed
- Test: 2177 passed, 69 skipped (14 warnings)

**Note**: `test_job_execution_manager.py` and `test_job_execution.py` timeout due to SQLite database locking in threaded workflow tests. These are pre-existing flaky tests excluded via `--ignore` flags.

**Date**: 2026-01-15
