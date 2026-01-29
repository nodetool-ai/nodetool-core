# datetime.utcnow Deprecation Fix

**Problem**: The `datetime.utcnow()` method is deprecated in Python 3.12 and will be removed in Python 3.14. It produces naive (timezone-unaware) datetime objects, which can lead to bugs in deployments across different timezones.

**Solution**: Replaced all occurrences of `datetime.utcnow()` with `datetime.now(UTC)` using Python 3.11's `datetime.UTC` constant.

**Changes**:
- `src/nodetool/migrations/runner.py`:
  - Line 271: `datetime.utcnow().isoformat()` → `datetime.now(UTC).isoformat()`
  - Line 319: `datetime.utcnow().isoformat()` → `datetime.now(UTC).isoformat()`
  - Line 331: `datetime.utcnow()` → `datetime.now(UTC)`
  - Line 340: `datetime.utcnow().isoformat()` → `datetime.now(UTC).isoformat()`
  - Updated import: `from datetime import UTC, datetime`

**Why**: 
- `datetime.now(UTC)` returns a timezone-aware datetime in UTC
- Using `UTC` constant is more idiomatic in Python 3.11+
- Avoids deprecation warnings and potential bugs

**Files**: `src/nodetool/migrations/runner.py`

**Date**: 2026-01-16
