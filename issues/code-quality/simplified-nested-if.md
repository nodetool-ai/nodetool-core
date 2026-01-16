# Code Quality - Simplified Nested If Statements

**Problem**: Multiple files contained unnecessarily nested if statements that reduced code readability and increased cyclomatic complexity.

**Solution**: Simplified nested if statements using logical `and` operators in the following locations:

1. `src/nodetool/agents/tools/model_tools.py:103-109` - Combined query filter conditions
2. `src/nodetool/api/job.py:409-414` - Combined status check conditions
3. `src/nodetool/chat/search_nodes.py:116-121` - Combined type matching conditions
4. `src/nodetool/chat/search_nodes.py:124-126` - Combined regex match conditions

**Before**:
```python
if query:
    if (
        query not in m.id.lower()
        and query not in m.name.lower()
        and query not in (m.description or "").lower()
    ):
        continue
```

**After**:
```python
if query and (
    query not in m.id.lower()
    and query not in m.name.lower()
    and query not in (m.description or "").lower()
):
    continue
```

**Impact**:
- Improved code readability
- Reduced indentation levels
- Lower cyclomatic complexity
- Easier to understand control flow

**Date**: 2026-01-16
