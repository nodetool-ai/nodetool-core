# February 2026 Code Quality Fixes

**Problem**: Multiple linting and type safety issues across the codebase

**Solution**:
1. Fixed type annotation quotes in `api/server.py` (UP037) - removed unnecessary quotes from type hints
2. Fixed `click.prompt()` None type issues in `cli.py` - added assertions to guarantee non-None values
3. Fixed deprecated `datetime.utcnow()` in `security/user_manager.py` - replaced with `datetime.now(timezone.utc)`
4. Fixed SSH module type annotations in `deploy/ssh.py` - added `Any` type annotations for conditional imports
5. Fixed trailing whitespace in `cli.py` (W293)
6. Auto-fixed 39 additional linting issues with ruff

**Why**:
- Type annotation quotes are unnecessary in Python 3.11+ when types are already available
- `click.prompt()` returns `str | None` but `APIUserManager.__init__()` requires `str`
- `datetime.utcnow()` is deprecated in Python 3.12+ and will be removed in Python 3.14
- Conditional imports assigned to `None` need explicit type annotations to satisfy type checkers
- Trailing whitespace causes W293 lint warnings

**Files**:
- `src/nodetool/api/server.py`
- `src/nodetool/cli.py`
- `src/nodetool/security/user_manager.py`
- `src/nodetool/deploy/ssh.py`
- Multiple test files (auto-fixed)

**Impact**:
- Reduced typecheck warnings from 26 to 15 (42% reduction)
- Reduced lint errors from 62 to 48 (23% reduction)
- All error-level typecheck issues eliminated
- 39 additional lint issues auto-fixed

**Date**: 2026-02-13
