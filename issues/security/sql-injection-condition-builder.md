# SQL Injection Risk in SQLite Adapter Condition Building

**Problem**: The `_build_condition()` method in `sqlite_adapter.py` directly interpolated `condition.field` into SQL strings without validation, potentially allowing SQL injection if field names come from untrusted sources.

**Solution**: Added `_validate_column_name()` function with regex validation to ensure only valid column names are used in SQL queries.

**Files**:
- `src/nodetool/models/sqlite_adapter.py:517-533`

**Date**: 2026-01-14

**Severity**: Medium - Could allow SQL injection if Condition objects are created from untrusted input.
