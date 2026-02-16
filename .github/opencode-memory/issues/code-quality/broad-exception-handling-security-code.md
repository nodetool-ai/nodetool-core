# Broad Exception Handling in Security Code

**Problem**: Security-critical code used overly broad `except Exception:` handlers that could mask serious bugs or security vulnerabilities.

**Solution**: Replace with specific exception types that can actually occur in the code paths.

**Why**: Broad exception handlers in security code can hide encryption/decryption failures, database errors, or other critical issues that should be visible to operators.

**Files**:
- `src/nodetool/security/crypto.py:115` - Fixed in `is_valid_master_key()`
- `src/nodetool/security/secret_helper.py:345` - Fixed in `has_secret()`

**Date**: 2026-02-16
