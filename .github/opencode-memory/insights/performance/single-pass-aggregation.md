# Single-Pass Aggregation Pattern

**Problem**: Multiple separate iterations over the same collection for aggregation calculations.

**File**: `src/nodetool/models/prediction.py:235-238`

**Original Code**:
```python
total_cost = sum(p.cost or 0 for p in predictions)
total_input_tokens = sum(p.input_tokens or 0 for p in predictions)
total_output_tokens = sum(p.output_tokens or 0 for p in predictions)
total_tokens = sum(p.total_tokens or 0 for p in predictions)
# 4 separate iterations over predictions
```

**Solution**: Use a single loop to accumulate all values:
```python
total_cost = 0
total_input_tokens = 0
total_output_tokens = 0
total_tokens = 0
for p in predictions:
    total_cost += p.cost or 0
    total_input_tokens += p.input_tokens or 0
    total_output_tokens += p.output_tokens or 0
    total_tokens += p.total_tokens or 0
# Single iteration
```

**Impact**: Reduces iterations from O(4n) to O(n), improving memory efficiency and cache locality.

**Date**: 2026-01-16
