from unittest.mock import MagicMock

import pytest

from nodetool.agents import task_executor
from nodetool.agents.task_executor import TaskExecutor
from nodetool.metadata.types import Step, Task
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


class DummyStepExecutor:
    """Simple StepExecutor replacement used for testing."""

    def __init__(self, *, task, step, processing_context, **kwargs):
        self.step = step

    async def execute(self):
        self.step.completed = True
        yield Chunk(content=f"done:{self.step.id}", done=True)


async def dummy_wrap_generators_parallel(*generators):
    for gen in generators:
        async for item in gen:
            yield item


def make_executor(tmp_path, parallel=False):
    context = ProcessingContext(workspace_dir=str(tmp_path))
    task = Task(title="t", steps=[])
    provider = MagicMock()
    return (
        TaskExecutor(
            provider=provider,
            model="model",
            processing_context=context,
            tools=[],
            task=task,
            parallel_execution=parallel,
        ),
        context,
    )


def test_get_all_executable_tasks(tmp_path):
    context = ProcessingContext(workspace_dir=str(tmp_path))

    s1 = Step(instructions="a")
    s2 = Step(instructions="b", completed=True)
    s3 = Step(instructions="c")
    s4 = Step(instructions="d", depends_on=["missing_task"])
    s5 = Step(instructions="e")
    s5.start_time = 1  # Running

    task = Task(title="t", steps=[s1, s2, s3, s4, s5])
    executor = TaskExecutor(MagicMock(), "model", context, [], task)

    exec_tasks = executor._get_all_executable_tasks()
    # s1 and s3 are executable (not completed, not running, no unmet dependencies)
    # s2 is completed, s4 has unmet dependency, s5 is running
    assert exec_tasks == [s1, s3]


def test_all_tasks_complete(tmp_path):
    s1 = Step(instructions="a")
    s2 = Step(instructions="b", completed=True)
    task = Task(title="t", steps=[s1, s2])
    executor = TaskExecutor(
        MagicMock(), "m", ProcessingContext(workspace_dir=str(tmp_path)), [], task
    )

    assert not executor._all_tasks_complete()
    s1.completed = True
    assert executor._all_tasks_complete()


@pytest.mark.asyncio
async def test_execute_tasks_sequential(monkeypatch, tmp_path):
    sub1 = Step(instructions="a")
    sub2 = Step(instructions="b")
    task = Task(title="t", steps=[sub1, sub2])
    context = ProcessingContext(workspace_dir=str(tmp_path))
    executor = TaskExecutor(
        MagicMock(), "m", context, [], task, parallel_execution=False
    )

    monkeypatch.setattr(task_executor, "StepExecutor", DummyStepExecutor)

    results = []
    async for item in executor.execute_tasks(context):
        results.append(item)

    assert [sub1.completed, sub2.completed] == [True, True]
    assert all(isinstance(r, Chunk) for r in results)


@pytest.mark.asyncio
async def test_execute_tasks_parallel(monkeypatch, tmp_path):
    sub1 = Step(instructions="a")
    sub2 = Step(instructions="b")
    task = Task(title="t", steps=[sub1, sub2])
    context = ProcessingContext(workspace_dir=str(tmp_path))
    executor = TaskExecutor(
        MagicMock(), "m", context, [], task, parallel_execution=True
    )

    monkeypatch.setattr(task_executor, "StepExecutor", DummyStepExecutor)
    monkeypatch.setattr(
        task_executor, "wrap_generators_parallel", dummy_wrap_generators_parallel
    )

    called = []

    async for item in executor.execute_tasks(context):
        called.append(item)

    assert sub1.completed and sub2.completed
    assert len(called) == 2
