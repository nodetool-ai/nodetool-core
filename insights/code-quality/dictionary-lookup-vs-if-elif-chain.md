# Dictionary Lookup vs If-Elif Chain Pattern

**Insight**: When mapping keys to values in Python, using a dictionary lookup is more efficient and more maintainable than consecutive if-elif statements.

**Rationale**: Dictionaries provide O(1) average case lookup time, whereas if-elif chains require O(n) comparisons. Dictionaries are also more Pythonic and easier to extend.

**Example**:
```python
# Before (less efficient, harder to maintain)
async def get_secret_mock(key):
    if key == "APIFY_API_KEY":
        return "test_apify_key"
    elif key == "DATA_FOR_SEO_LOGIN":
        return "test_login"
    elif key == "DATA_FOR_SEO_PASSWORD":
        return "test_password"
    return None

# After (more efficient, easier to maintain)
async def get_secret_mock(key):
    secret_map = {
        "APIFY_API_KEY": "test_apify_key",
        "DATA_FOR_SEO_LOGIN": "test_login",
        "DATA_FOR_SEO_PASSWORD": "test_password",
    }
    return secret_map.get(key)
```

**Impact**:
- Better performance for multiple keys (O(1) vs O(n))
- Easier to add/remove mappings
- More concise and readable code
- Follows Python best practices (ruff SIM116 rule)

**Files**:
- `tests/agents/tools/test_serp_provider_selection.py`

**Date**: 2026-02-15
