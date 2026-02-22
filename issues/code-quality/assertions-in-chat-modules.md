# Assertions Replaced with Proper Exceptions in Chat Modules

**Problem**: Multiple chat modules used `assert` statements for runtime validation, which is problematic because:
1. Python's `-O` flag optimizes away assert statements, silently skipping validation
2. AssertionError doesn't clearly indicate what went wrong
3. Production code should use explicit exception types for proper error handling

**Solution**: Replaced `assert` statements with proper exception types:
- `ValueError` for missing/invalid parameters
- `TypeError` for type mismatches
- `RuntimeError` for internal state errors

**Why**: Using proper exceptions ensures that validation always runs regardless of Python optimization flags, provides clearer error messages, and allows callers to catch specific exception types for proper error handling.

**Files**:
- `src/nodetool/chat/regular_chat.py:85,164`
- `src/nodetool/chat/base_chat_runner.py:318,434,445-446,484-485`
- `src/nodetool/chat/chat.py:51`
- `src/nodetool/cli_migrations.py:84`

**Date**: 2026-02-21
