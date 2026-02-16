# Exception Handling in Security Code

**Insight**: Security-critical code must use specific exception types rather than broad `except Exception:` handlers to ensure failures are visible and actionable.

**Rationale**:
- Broad exception handlers can mask encryption/decryption failures
- Database errors in security code should be visible to operators
- Specific exceptions make debugging and monitoring easier
- Prevents silent failures that could expose security vulnerabilities

**Example**:
```python
# Bad - masks all errors
try:
    SecretCrypto.decrypt(test_encrypted_value, master_key, user_id)
    return True
except Exception:
    return False

# Good - handles only expected errors
try:
    SecretCrypto.decrypt(test_encrypted_value, master_key, user_id)
    return True
except (InvalidToken, ValueError, UnicodeDecodeError):
    # InvalidToken: Wrong key or corrupted data
    # ValueError: Decryption failed (raised by decrypt())
    # UnicodeDecodeError: Decrypted bytes are not valid UTF-8
    return False
```

**Impact**: Improved observability and debuggability in security-critical code paths.

**Files**: `src/nodetool/security/crypto.py`, `src/nodetool/security/secret_helper.py`

**Date**: 2026-02-16
