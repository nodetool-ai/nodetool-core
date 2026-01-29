# Enhanced AsyncTaskGroup with Priorities and Deadlines

**Insight**: Extended AsyncTaskGroup with task priorities and per-task deadlines for better workflow execution control.

**Feature**: New experimental module at `src/nodetool/concurrency/task_group_enhanced.py` with:
- `priority` parameter on spawn (lower = higher priority, 0-100)
- `deadline` parameter on spawn (seconds before task is cancelled)
- `depends_on` parameter for task dependencies
- `TaskPriority` class for priority ordering
- `TaskDeadlineExceededError` exception
- Updated `TaskStats` with `deadline_exceeded` count
- `priorities` property for getting task priorities
- `get_deadline()` method for checking task deadlines
- `is_dependency_satisfied()` method for checking dependency completion

**Rationale**: The existing AsyncTaskGroup lacked:
1. Task prioritization for ordering task execution
2. Per-task timeouts (only had group-level timeout)
3. Dependency tracking between tasks

**Example**:
```python
from nodetool.concurrency.task_group_enhanced import (
    AsyncTaskGroup, PRIORITY_HIGH, PRIORITY_LOW
)

group = AsyncTaskGroup()
group.spawn("critical", critical_task(), priority=PRIORITY_HIGH, deadline=5.0)
group.spawn("background", background_task(), priority=PRIORITY_LOW)

results = await group.run()
```

**Impact**:
- Better control over task execution order
- Prevents long-running tasks from blocking workflows
- Enables DAG-like task dependencies
- Backwards compatible with existing AsyncTaskGroup API

**Trade-offs**:
- Deadline exceeded is treated as a failure (like cancellation)
- Dependencies don't automatically delay task start (events are set on completion)
- Priority ordering is available via `_get_execution_order()` but not enforced during concurrent execution

**Files**:
- `src/nodetool/concurrency/task_group_enhanced.py`
- `tests/concurrency/test_task_group_enhanced.py`

**Date**: 2026-01-22
