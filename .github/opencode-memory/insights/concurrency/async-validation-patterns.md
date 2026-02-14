# Async Validation Patterns

**Insight**: Async validation utilities fill the gap between synchronous type checking and nodetool's async-first architecture, providing extensible validation patterns for nodes, workflows, and providers.

**Rationale**: While nodetool has excellent synchronous type checking (`typecheck()`, `is_assignable()` in `metadata/typecheck.py`), many validation scenarios require async operations (network checks, database lookups, external API calls). The new async validation utilities provide:

1. **AsyncValidator base class** - Easy-to-extend base for custom async validators
2. **ValidationResult dataclass** - Structured results with errors, warnings, and context
3. **validate_async()** - Run multiple validators with configurable error handling
4. **validate_with_retries()** - Retry validation for transient failures
5. **ValidatorComposer** - Compose validators with AND logic using fluent API
6. **AnyValidatorComposer** - Compose validators with OR logic (at least one must pass)
7. **ConditionalValidator** - Apply validation conditionally based on predicates

**Example**:
```python
# Define custom validators
class URLValidator(AsyncValidator):
    async def validate_async(self, value):
        if not value.startswith("http"):
            return ValidationResult(False, errors=["Invalid URL"])
        return ValidationResult(True)

class LengthValidator(AsyncValidator):
    def __init__(self, min_len=0, max_len=1000):
        self.min_len = min_len
        self.max_len = max_len
    
    async def validate_async(self, value):
        if len(value) < self.min_len or len(value) > self.max_len:
            return ValidationResult(False, errors=["Length out of range"])
        return ValidationResult(True)

# Use composition
validator = ValidatorComposer.all_of([
    URLValidator(),
    LengthValidator(min_len=10, max_len=500)
])

result = await validator.validate_async("https://example.com")
```

**Impact**: Provides consistent async validation patterns across the codebase, reducing code duplication and improving error handling in async contexts.

**Files**:
- `src/nodetool/concurrency/validation.py` - Main implementation
- `tests/concurrency/test_validation.py` - Comprehensive test suite (27 tests)
- `src/nodetool/concurrency/__init__.py` - Exports validation utilities

**Date**: 2026-02-13
