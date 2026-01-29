import asyncio

import pytest

from nodetool.concurrency.task_group_enhanced import (
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    AsyncTaskGroup,
    TaskDeadlineExceededError,
    TaskExecutionError,
    TaskPriority,
    TaskResult,
)


@pytest.mark.asyncio
async def test_task_priority_class():
    """Test TaskPriority ordering."""
    low = TaskPriority(PRIORITY_LOW)
    high = TaskPriority(PRIORITY_HIGH)
    normal = TaskPriority(PRIORITY_NORMAL)

    assert high.value < normal.value < low.value


@pytest.mark.asyncio
async def test_task_deadline_exceeded():
    """Test that tasks exceeding their deadline are marked as such."""

    async def slow_task() -> str:
        await asyncio.sleep(0.5)
        return "done"

    async with AsyncTaskGroup() as group:
        group.spawn("slow", slow_task(), deadline=0.1)

    results = group.results
    assert len(results) == 1
    assert results[0].deadline_exceeded
    assert results[0].task_id == "slow"


@pytest.mark.asyncio
async def test_task_deadline_completed():
    """Test that tasks completing within deadline succeed."""

    async def fast_task() -> str:
        await asyncio.sleep(0.05)
        return "done"

    async with AsyncTaskGroup() as group:
        group.spawn("fast", fast_task(), deadline=0.5)

    results = group.results
    assert len(results) == 1
    assert results[0].success
    assert results[0].result == "done"


@pytest.mark.asyncio
async def test_stats_include_deadline_exceeded():
    """Test that stats include deadline_exceeded count."""

    async def slow_task() -> str:
        await asyncio.sleep(0.5)
        return "done"

    async with AsyncTaskGroup() as group:
        group.spawn("slow", slow_task(), deadline=0.1)

    stats = group.stats
    assert stats.deadline_exceeded == 1
    assert stats.total == 1


@pytest.mark.asyncio
async def test_get_deadline():
    """Test getting deadline for a task."""
    async with AsyncTaskGroup() as group:
        group.spawn("with_deadline", asyncio.sleep(0), deadline=5.0)
        group.spawn("without_deadline", asyncio.sleep(0))

    assert group.get_deadline("with_deadline") == 5.0
    assert group.get_deadline("without_deadline") is None


@pytest.mark.asyncio
async def test_priorities_property():
    """Test getting priorities for all tasks."""
    async with AsyncTaskGroup() as group:
        group.spawn("high", asyncio.sleep(0), priority=PRIORITY_HIGH)
        group.spawn("low", asyncio.sleep(0), priority=PRIORITY_LOW)

    priorities = group.priorities
    assert priorities["high"] == PRIORITY_HIGH
    assert priorities["low"] == PRIORITY_LOW


@pytest.mark.asyncio
async def test_error_on_duplicate_task_id():
    """Test that spawning a duplicate task ID raises an error."""
    async with AsyncTaskGroup() as group:
        group.spawn("task1", asyncio.sleep(0))
        with pytest.raises(ValueError, match="already exists"):
            group.spawn("task1", asyncio.sleep(0))


@pytest.mark.asyncio
async def test_run_with_raise_on_error():
    """Test run method respects raise_on_error flag."""

    async def failing_task() -> str:
        raise ValueError("Task failed")

    group = AsyncTaskGroup()
    group.spawn("fail", failing_task())

    with pytest.raises(TaskExecutionError):
        await group.run(raise_on_error=True)


@pytest.mark.asyncio
async def test_run_without_raise_on_error():
    """Test run method returns results even with failures when raise_on_error=False."""

    async def failing_task() -> str:
        raise ValueError("Task failed")

    group = AsyncTaskGroup()
    group.spawn("fail", failing_task())

    results = await group.run(raise_on_error=False)

    assert len(results) == 1
    assert not results[0].success
    assert isinstance(results[0].exception, ValueError)


@pytest.mark.asyncio
async def test_backwards_compatibility():
    """Test that original API still works without new parameters."""
    results_list: list[str] = []

    async def task(name: str) -> str:
        results_list.append(name)
        return name

    async with AsyncTaskGroup() as group:
        group.spawn("a", task("a"))
        group.spawn("b", task("b"))

    results = group.results

    assert len(results) == 2
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_cancel_on_error():
    """Test that cancel_on_error cancels remaining tasks."""

    async def slow_task() -> str:
        await asyncio.sleep(10.0)
        return "done"

    async def failing_task() -> str:
        await asyncio.sleep(0.01)
        raise ValueError("Task failed")

    group = AsyncTaskGroup()
    group.spawn("slow", slow_task())
    group.spawn("fail", failing_task())

    with pytest.raises(TaskExecutionError):
        await group.run(cancel_on_error=True)

    assert group.pending == []


@pytest.mark.asyncio
async def test_deadline_causes_failure():
    """Test that deadline exceeded causes a failed result."""

    async def slow_task() -> str:
        await asyncio.sleep(0.5)
        return "done"

    group = AsyncTaskGroup()
    group.spawn("slow", slow_task(), deadline=0.1)

    results = await group.run(raise_on_error=False)
    assert len(results) == 1
    assert not results[0].success
    assert results[0].deadline_exceeded


@pytest.mark.asyncio
async def test_context_manager_cancels_on_error():
    """Test that context manager properly cancels tasks on errors."""

    async def failing_task() -> str:
        raise ValueError("Task failed")

    async with AsyncTaskGroup() as group:
        group.spawn("fail", failing_task())
        group.spawn("other", asyncio.sleep(1.0))

    assert group.pending == []


@pytest.mark.asyncio
async def test_get_execution_order():
    """Test that execution order is sorted by priority."""
    async with AsyncTaskGroup() as group:
        group.spawn("low", asyncio.sleep(0), priority=PRIORITY_LOW)
        group.spawn("high", asyncio.sleep(0), priority=PRIORITY_HIGH)
        group.spawn("normal", asyncio.sleep(0), priority=PRIORITY_NORMAL)

    order = group._get_execution_order()
    assert order == ["high", "normal", "low"]
