# Unused Unpacked Variables

**Problem**: RUF059 linting errors - test code unpacked tuple values that were never used, causing ruff to warn about dead code.

**Solution**: Prefixed unused unpacked variables with underscore in:
- `tests/ui/test_console_module.py`: Changed `label` and `is_error` to `_label` and `_is_error` (except where `label` is used)
- `tests/workflows/test_actor.py`: Changed `runner` and `ctx` to `_runner` and `_ctx`

**Why**: Prefixing with underscore signals to readers and linters that these variables are intentionally unused for structural unpacking purposes.

**Files**:
- `tests/ui/test_console_module.py`
- `tests/workflows/test_actor.py`

**Date**: 2026-02-07
