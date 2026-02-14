# Debug API Test Coverage

**Insight**: Added comprehensive test coverage for the debug API endpoints, focusing on security-critical secret redaction and export functionality.

**Rationale**: The debug API (`/api/debug/export`) creates ZIP bundles containing logs, workflows, and system information for troubleshooting. This functionality requires careful testing to ensure:
1. Secrets are properly redacted before sharing
2. Export bundles contain all expected files
3. System environment detection works correctly

**Example**:
```python
def test_redact_api_key_values(self):
    """Test that API keys are redacted."""
    data = {"api_key": "sk-test1234567890abcdefghijklmnop"}
    result = _redact_secrets(data)
    assert result["api_key"] == "[REDACTED]"
```

**Impact**:
- Added 17 tests covering secret redaction, version detection, save directory logic, and export endpoint
- Tests verify that API keys, tokens, and passwords are redacted from debug bundles
- Ensures safe keys (id, name, etc.) are preserved for debugging

**Files**:
- tests/api/test_debug.py
- src/nodetool/api/debug.py

**Date**: 2026-02-14
