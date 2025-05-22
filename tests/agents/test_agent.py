import json
import pytest

from nodetool.agents.agent import Agent
from nodetool.chat.providers.base import MockProvider
from nodetool.metadata.types import SubTask, Task, ToolCall
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import TaskUpdate, TaskUpdateEvent, SubTaskResult


@pytest.mark.asyncio
async def test_execute_with_initial_task(monkeypatch, tmp_path):
    subtask = SubTask(id="sub1", content="do", output_file="result.txt")
    task = Task(title="t", subtasks=[subtask])
    provider = MockProvider([])

    async def fake_execute_tasks(self, ctx):
        sub = self.task.subtasks[0]
        yield TaskUpdate(task=self.task, subtask=sub, event=TaskUpdateEvent.SUBTASK_STARTED)
        yield ToolCall(id="1", name="finish_subtask", args={"result": "part"}, subtask_id=sub.id)
        yield TaskUpdate(task=self.task, subtask=sub, event=TaskUpdateEvent.SUBTASK_COMPLETED)
        yield ToolCall(id="2", name="finish_task", args={"result": "final"}, subtask_id=sub.id)

    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            self.task = kwargs["task"]

        execute_tasks = fake_execute_tasks

    monkeypatch.setattr("nodetool.agents.agent.TaskExecutor", DummyExecutor)
    monkeypatch.setattr("nodetool.common.settings.get_log_path", lambda f: tmp_path / f)

    agent = Agent(
        name="my agent",
        objective="obj",
        provider=provider,
        model="m",
        task=task,
        output_type="text",
        output_schema={"type": "string"},
        verbose=False,
    )

    context = ProcessingContext(workspace_dir=str(tmp_path))
    results = []
    async for item in agent.execute(context):
        results.append(item)

    assert agent.get_results() == "final"
    assert subtask.output_file == str(tmp_path / "result.txt")
    assert subtask.output_type == "text"
    assert subtask.output_schema == json.dumps({"type": "string"})
    assert provider.log_file.endswith(f"__my_agent__{subtask.id}.jsonl")

    assert [type(i) for i in results] == [TaskUpdate, SubTaskResult, TaskUpdate, TaskUpdate]
    assert results[-1].event == TaskUpdateEvent.TASK_COMPLETED
