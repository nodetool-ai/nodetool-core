from unittest.mock import MagicMock

import pytest

from nodetool.agents import task_executor
from nodetool.agents.task_executor import TaskExecutor
from nodetool.metadata.types import SubTask, Task
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


class DummySubTaskContext:
    """Simple SubTaskContext replacement used for testing."""

    def __init__(self, *, task, subtask, processing_context, **kwargs):
        self.subtask = subtask

    async def execute(self):
        self.subtask.completed = True
        yield Chunk(content=f"done:{self.subtask.id}", done=True)


async def dummy_wrap_generators_parallel(*generators):
    for gen in generators:
        async for item in gen:
            yield item


def make_executor(tmp_path, parallel=False):
    context = ProcessingContext(workspace_dir=str(tmp_path))
    task = Task(title="t", subtasks=[])
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


def test_check_input_files(tmp_path):
    executor, _ = make_executor(tmp_path)

    existing = tmp_path / "in.txt"
    existing.write_text("data")

    assert executor._check_input_files(["in.txt"], str(tmp_path))
    assert not executor._check_input_files(["missing.txt"], str(tmp_path))


def test_get_all_executable_tasks(tmp_path):
    context = ProcessingContext(workspace_dir=str(tmp_path))
    file_ok = tmp_path / "ok.txt"
    file_ok.write_text("data")

    s1 = SubTask(content="a", output_file="a.txt")
    s2 = SubTask(content="b", output_file="b.txt", completed=True)
    s3 = SubTask(content="c", output_file="c.txt", input_files=["ok.txt"])
    s4 = SubTask(content="d", output_file="d.txt", input_files=["missing.txt"])
    s5 = SubTask(content="e", output_file="e.txt")
    s5.start_time = 1

    task = Task(title="t", subtasks=[s1, s2, s3, s4, s5])
    executor = TaskExecutor(MagicMock(), "model", context, [], task)

    exec_tasks = executor._get_all_executable_tasks()
    assert exec_tasks == [s1, s3]


def test_all_tasks_complete(tmp_path):
    s1 = SubTask(content="a", output_file="a.txt")
    s2 = SubTask(content="b", output_file="b.txt", completed=True)
    task = Task(title="t", subtasks=[s1, s2])
    executor = TaskExecutor(
        MagicMock(), "m", ProcessingContext(workspace_dir=str(tmp_path)), [], task
    )

    assert not executor._all_tasks_complete()
    s1.completed = True
    assert executor._all_tasks_complete()


@pytest.mark.asyncio
async def test_execute_tasks_sequential(monkeypatch, tmp_path):
    sub1 = SubTask(content="a", output_file="a.txt")
    sub2 = SubTask(content="b", output_file="b.txt")
    task = Task(title="t", subtasks=[sub1, sub2])
    context = ProcessingContext(workspace_dir=str(tmp_path))
    executor = TaskExecutor(
        MagicMock(), "m", context, [], task, parallel_execution=False
    )

    monkeypatch.setattr(task_executor, "SubTaskContext", DummySubTaskContext)

    results = []
    async for item in executor.execute_tasks(context):
        results.append(item)

    assert [sub1.completed, sub2.completed] == [True, True]
    assert all(isinstance(r, Chunk) for r in results)


@pytest.mark.asyncio
async def test_execute_tasks_parallel(monkeypatch, tmp_path):
    sub1 = SubTask(content="a", output_file="a.txt")
    sub2 = SubTask(content="b", output_file="b.txt")
    task = Task(title="t", subtasks=[sub1, sub2])
    context = ProcessingContext(workspace_dir=str(tmp_path))
    executor = TaskExecutor(
        MagicMock(), "m", context, [], task, parallel_execution=True
    )

    monkeypatch.setattr(task_executor, "SubTaskContext", DummySubTaskContext)
    monkeypatch.setattr(
        task_executor, "wrap_generators_parallel", dummy_wrap_generators_parallel
    )

    called = []

    async for item in executor.execute_tasks(context):
        called.append(item)

    assert sub1.completed and sub2.completed
    assert len(called) == 2
