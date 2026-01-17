# Empty Exception Classes

**Problem**: Exception classes in `migrations/exceptions.py` had unnecessary `pass` statements when they only contained docstrings.

**Solution**: Removed `pass` statements from `LockError`, `BaselineError`, `MigrationDiscoveryError`, and `RollbackError` classes. Empty exception classes with only docstrings don't need `pass` statements.

**Why**: Python allows class bodies to be empty (docstring-only), and removing unnecessary `pass` reduces visual noise.

**Files**: `src/nodetool/migrations/exceptions.py`

**Date**: 2026-01-17
