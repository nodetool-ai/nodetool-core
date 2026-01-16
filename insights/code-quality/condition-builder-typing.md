# ConditionBuilder List Type Variance

**Insight**: The `ConditionGroup` constructor expects `List[ConditionGroup | Condition | ConditionBuilder]` but when building lists dynamically with list comprehensions, Python's list type is invariant. This causes type checker errors even though the code works correctly at runtime because `ConditionGroup.__init__` calls `.build()` on `ConditionBuilder` objects.

**Example**:
```python
conditions = [Field(k).equals(v) for k, v in query.items()]
condition = ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND))  # type: ignore[arg-type]
```

**Impact**: The type ignore comment is necessary and consistent with existing patterns in the codebase.

**Files**:
- `src/nodetool/models/run_inbox_message.py`

**Date**: 2026-01-16
