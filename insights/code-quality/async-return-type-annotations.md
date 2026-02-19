# Async Return Type Annotations

**Insight**: All async functions and methods should have explicit return type annotations, even when they return `None`.

**Rationale**: 
- Type checkers (basedpyright, mypy) require explicit return types for proper type inference
- Python 3.11+ best practices recommend explicit type annotations for all public APIs
- Async functions without return types can hide bugs where return values are incorrectly used
- Makes code intent clear to readers and tools

**Example**:
```python
# Before (implicit None return)
async def handle_message(self, messages: list[ApiMessage]):
    await self.process(messages)

# After (explicit None return)
async def handle_message(self, messages: list[ApiMessage]) -> None:
    await self.process(messages)
```

**Impact**: Adding return type annotations helps type checkers catch bugs where return values are incorrectly expected or used. For example, if a function is changed to return a value but callers don't expect it, the type checker will flag the issue.

**Files**: All async functions in `src/nodetool/` should have explicit return types

**Date**: 2026-02-19
