import asyncio

import pytest

from nodetool.concurrency.async_task_group import (
    AsyncTaskGroup,
    AsyncTaskGroupError,
    TaskExecutionError,
    TaskResult,
    TaskStats,
)


@pytest.fixture
def event_loop():
    """Create an event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_spawn_single_task():
    """Test spawning and executing a single task."""

    async def simple_task():
        return "result"

    group = AsyncTaskGroup()
    group.spawn("task1", simple_task())
    results = await group.run()

    assert len(results) == 1
    assert results[0].task_id == "task1"
    assert results[0].result == "result"
    assert results[0].success is True


@pytest.mark.asyncio
async def test_spawn_multiple_tasks():
    """Test spawning and executing multiple tasks."""

    async def task_with_value(value: int):
        return value * 2

    group = AsyncTaskGroup()
    group.spawn("task1", task_with_value(1))
    group.spawn("task2", task_with_value(2))
    group.spawn("task3", task_with_value(3))

    results = await group.run()

    assert len(results) == 3
    results_dict = {r.task_id: r for r in results}
    assert results_dict["task1"].result == 2
    assert results_dict["task2"].result == 4
    assert results_dict["task3"].result == 6


@pytest.mark.asyncio
async def test_task_exception_handling():
    """Test that exceptions in tasks are captured correctly."""

    async def failing_task():
        raise ValueError("Task failed")

    group = AsyncTaskGroup()
    group.spawn("failing", failing_task())
    results = await group.run(raise_on_error=False)

    assert len(results) == 1
    assert results[0].success is False
    assert isinstance(results[0].exception, ValueError)
    assert str(results[0].exception) == "Task failed"


@pytest.mark.asyncio
async def test_raise_on_error():
    """Test that TaskExecutionError is raised when configured."""

    async def failing_task():
        raise RuntimeError("Error")

    async def success_task():
        await asyncio.sleep(0.1)
        return "ok"

    group = AsyncTaskGroup()
    group.spawn("task1", failing_task())
    group.spawn("task2", success_task())

    with pytest.raises(TaskExecutionError) as exc_info:
        await group.run(raise_on_error=True, cancel_on_error=False)

    assert len(exc_info.value.failed_results) == 1
    assert exc_info.value.failed_results[0].task_id == "task1"


@pytest.mark.asyncio
async def test_cancel_on_error():
    """Test that remaining tasks are cancelled when one fails."""

    async def slow_task():
        await asyncio.sleep(10.0)
        return "never"

    async def failing_task():
        await asyncio.sleep(0.1)
        raise ValueError("Fail")

    group = AsyncTaskGroup()
    group.spawn("slow", slow_task())
    group.spawn("failing", failing_task())

    start = asyncio.get_running_loop().time()
    with pytest.raises(TaskExecutionError):
        await group.run(raise_on_error=True, cancel_on_error=True)
    elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 2.0
    assert group.stats.cancelled == 1


@pytest.mark.asyncio
async def test_cancel_all():
    """Test manual cancellation of all tasks."""

    async def slow_task():
        await asyncio.sleep(10.0)
        return "slow"

    group = AsyncTaskGroup()
    group.spawn("task1", slow_task())
    group.spawn("task2", slow_task())

    await asyncio.sleep(0.1)
    await group.cancel_all()

    stats = group.stats
    assert stats.cancelled == 2


@pytest.mark.asyncio
async def test_run_until_first():
    """Test running until first task completes successfully."""

    async def slow_task():
        await asyncio.sleep(0.5)
        return "slow"

    async def fast_task():
        await asyncio.sleep(0.05)
        return "fast"

    async def medium_task():
        await asyncio.sleep(0.2)
        return "medium"

    group = AsyncTaskGroup()
    group.spawn("slow", slow_task())
    group.spawn("fast", fast_task())
    group.spawn("medium", medium_task())

    result = await group.run_until_first()

    assert result.success is True
    assert result.result == "fast"


@pytest.mark.asyncio
async def test_run_until_first_timeout():
    """Test timeout in run_until_first."""

    async def slow_task():
        await asyncio.sleep(10.0)
        return "never"

    group = AsyncTaskGroup()
    group.spawn("task1", slow_task())

    with pytest.raises(asyncio.TimeoutError):
        await group.run_until_first(timeout=0.1)


@pytest.mark.asyncio
async def test_pending_property():
    """Test the pending property returns correct task IDs."""

    async def fast_task():
        return "fast"

    async def slow_task():
        await asyncio.sleep(0.5)
        return "slow"

    group = AsyncTaskGroup()
    group.spawn("fast", fast_task())
    group.spawn("slow", slow_task())

    await asyncio.sleep(0.1)

    assert "slow" in group.pending
    assert "fast" not in group.pending


@pytest.mark.asyncio
async def test_stats_property():
    """Test the stats property returns correct counts."""

    async def success_task(value: int):
        return value

    async def fail_task():
        raise ValueError("Fail")

    group = AsyncTaskGroup()
    group.spawn("success1", success_task(1))
    group.spawn("success2", success_task(2))
    group.spawn("fail", fail_task())

    await group.run(raise_on_error=False)

    stats = group.stats
    assert stats.total == 3
    assert stats.completed == 2
    assert stats.failed == 1
    assert stats.cancelled == 0


@pytest.mark.asyncio
async def test_duplicate_task_id_raises():
    """Test that spawning a task with duplicate ID raises ValueError."""

    async def task():
        return "result"

    group = AsyncTaskGroup()
    group.spawn("task1", task())

    with pytest.raises(ValueError) as exc_info:
        group.spawn("task1", task())

    assert "already exists" in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_group():
    """Test running an empty task group."""
    group = AsyncTaskGroup()
    results = await group.run()

    assert results == []


@pytest.mark.asyncio
async def test_context_manager():
    """Test using AsyncTaskGroup as async context manager."""

    async def task():
        return "result"

    async with AsyncTaskGroup() as group:
        group.spawn("task1", task())
        group.spawn("task2", task())

    results = group.results
    assert len(results) == 2
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_context_manager_cancellation():
    """Test that context manager cancels tasks on exception."""

    async def slow_task():
        await asyncio.sleep(10.0)
        return "slow"

    async def fail_after_slow():
        await asyncio.sleep(0.1)
        raise RuntimeError("Error")

    with pytest.raises(RuntimeError):
        async with AsyncTaskGroup() as group:
            group.spawn("slow", slow_task())
            group.spawn("failing", fail_after_slow())

    stats = group.stats
    assert stats.cancelled == 1


@pytest.mark.asyncio
async def test_callbacks():
    """Test success and error callbacks."""
    success_values: list[int] = []
    error_values: list[BaseException] = []

    async def success_task(value: int):
        return value

    async def fail_task():
        raise ValueError("Fail")

    group = AsyncTaskGroup()
    group.spawn(
        "task1",
        success_task(1),
        on_success=lambda v: success_values.append(v),
    )
    group.spawn(
        "task2",
        fail_task(),
        on_error=lambda e: error_values.append(e),
    )

    await group.run(raise_on_error=False)

    assert success_values == [1]
    assert len(error_values) == 1
    assert isinstance(error_values[0], ValueError)


@pytest.mark.asyncio
async def test_already_running_raises():
    """Test that running a group twice raises an error."""

    async def task():
        return "result"

    group = AsyncTaskGroup()
    group.spawn("task", task())

    await group.run()

    with pytest.raises(AsyncTaskGroupError):
        await group.run()


@pytest.mark.asyncio
async def test_wait_method():
    """Test the wait method waits for all tasks without collecting."""

    async def slow_task():
        await asyncio.sleep(0.1)
        return "done"

    group = AsyncTaskGroup()
    group.spawn("task", slow_task())

    await group.wait()

    assert not group.pending


@pytest.mark.asyncio
async def test_mixed_success_and_failure():
    """Test a mix of successful and failed tasks."""

    async def succeed():
        return "ok"

    async def fail():
        raise RuntimeError("Failed")

    group = AsyncTaskGroup()
    group.spawn("success1", succeed())
    group.spawn("fail1", fail())
    group.spawn("success2", succeed())
    group.spawn("fail2", fail())

    results = await group.run(raise_on_error=False)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    assert len(successful) == 2
    assert len(failed) == 2
    assert all(r.result == "ok" for r in successful)
    assert all(isinstance(r.exception, RuntimeError) for r in failed)
