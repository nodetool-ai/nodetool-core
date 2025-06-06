import json
import pytest

from nodetool.agents.agent import Agent
from nodetool.chat.providers.base import MockProvider
from nodetool.metadata.types import SubTask, Task, TaskPlan, ToolCall
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import (
    Chunk,
    PlanningUpdate,
    TaskUpdate,
    TaskUpdateEvent,
    SubTaskResult,
)


@pytest.mark.asyncio
async def test_execute_with_initial_task(monkeypatch, tmp_path):
    subtask = SubTask(id="sub1", content="do", output_file="result.txt")
    task = Task(title="t", subtasks=[subtask])
    provider = MockProvider([])

    async def fake_execute_tasks(self, ctx):
        sub = self.task.subtasks[0]
        yield TaskUpdate(
            task=self.task, subtask=sub, event=TaskUpdateEvent.SUBTASK_STARTED
        )
        yield ToolCall(
            id="1", name="finish_subtask", args={"result": "part"}, subtask_id=sub.id
        )
        yield TaskUpdate(
            task=self.task, subtask=sub, event=TaskUpdateEvent.SUBTASK_COMPLETED
        )
        yield ToolCall(
            id="2", name="finish_task", args={"result": "final"}, subtask_id=sub.id
        )

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

    assert [type(i) for i in results] == [
        TaskUpdate,
        SubTaskResult,
        TaskUpdate,
        TaskUpdate,
    ]
    assert results[-1].event == TaskUpdateEvent.TASK_COMPLETED


class DummyTaskExecutor:
    def __init__(self, *, provider, model, processing_context, tools, task, **kwargs):
        self.task = task

    async def execute_tasks(self, context):
        subtask = self.task.subtasks[0]
        yield TaskUpdate(
            task=self.task, subtask=subtask, event=TaskUpdateEvent.SUBTASK_STARTED
        )
        yield Chunk(content="work")
        yield ToolCall(
            id="1", name="finish_subtask", args={"result": "42"}, subtask_id=subtask.id
        )
        yield ToolCall(
            id="2", name="finish_task", args={"result": "done"}, subtask_id=subtask.id
        )
        subtask.completed = True
        yield TaskUpdate(
            task=self.task, subtask=subtask, event=TaskUpdateEvent.SUBTASK_COMPLETED
        )


class DummyTaskPlanner:
    def __init__(
        self,
        *,
        provider,
        model,
        reasoning_model,
        objective,
        workspace_dir,
        execution_tools,
        input_files,
        output_schema,
        enable_analysis_phase,
        enable_data_contracts_phase,
        use_structured_output,
        verbose,
    ):
        sub = SubTask(content="do", output_file="out.txt")
        task = Task(title="t", subtasks=[sub])
        self.task_plan = TaskPlan(title="p", tasks=[task])

    async def create_task(self, processing_context, objective):
        yield PlanningUpdate(phase="plan", status="ok", content="start")

    async def save_task_plan(self):
        pass


@pytest.mark.asyncio
async def test_agent_execute_with_initial_task(monkeypatch, tmp_path):
    provider = MockProvider([])
    sub = SubTask(content="do", output_file="out.txt")
    task = Task(title="t", subtasks=[sub])
    monkeypatch.setattr("nodetool.agents.agent.TaskExecutor", DummyTaskExecutor)
    agent = Agent(
        name="a",
        objective="obj",
        provider=provider,
        model="m",
        tools=[],
        task=task,
        verbose=False,
    )
    context = ProcessingContext(workspace_dir=str(tmp_path))
    items = []
    async for item in agent.execute(context):
        items.append(item)

    assert any(isinstance(i, SubTaskResult) for i in items)
    assert any(
        isinstance(i, TaskUpdate) and i.event == TaskUpdateEvent.TASK_COMPLETED
        for i in items
    )
    assert agent.get_results() == "done"


@pytest.mark.asyncio
async def test_agent_planning_and_input_files(monkeypatch, tmp_path):
    provider = MockProvider([])
    input_file = tmp_path / "input.txt"
    input_file.write_text("data")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr("nodetool.agents.agent.TaskExecutor", DummyTaskExecutor)
    monkeypatch.setattr("nodetool.agents.agent.TaskPlanner", DummyTaskPlanner)

    agent = Agent(
        name="a",
        objective="obj",
        provider=provider,
        model="m",
        tools=[],
        input_files=[str(input_file)],
        verbose=False,
    )
    context = ProcessingContext(workspace_dir=str(workspace))
    items = []
    async for item in agent.execute(context):
        items.append(item)

    copied_path = workspace / "input.txt"
    assert copied_path.exists()
    assert agent.task is not None
    assert any(isinstance(i, PlanningUpdate) for i in items)
    assert any(
        isinstance(i, TaskUpdate) and i.event == TaskUpdateEvent.TASK_CREATED
        for i in items
    )
    assert any(
        isinstance(i, TaskUpdate) and i.event == TaskUpdateEvent.TASK_COMPLETED
        for i in items
    )
    assert agent.get_results() == "done"
