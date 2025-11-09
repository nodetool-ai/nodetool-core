import json
import pytest

from nodetool.agents.agent import Agent
from nodetool.providers.base import MockProvider
from nodetool.metadata.types import SubTask, Task, TaskPlan
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
    subtask = SubTask(id="sub1", content="do")
    task = Task(title="t", subtasks=[subtask])
    provider = MockProvider([])

    async def fake_execute_tasks(self, ctx):
        sub = self.task.subtasks[0]
        yield TaskUpdate(
            task=self.task, subtask=sub, event=TaskUpdateEvent.SUBTASK_STARTED
        )
        yield SubTaskResult(subtask=sub, result="part", is_task_result=False)
        yield TaskUpdate(
            task=self.task, subtask=sub, event=TaskUpdateEvent.SUBTASK_COMPLETED
        )
        yield SubTaskResult(subtask=sub, result="final", is_task_result=True)

    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            self.task = kwargs["task"]

        execute_tasks = fake_execute_tasks

    monkeypatch.setattr("nodetool.agents.agent.TaskExecutor", DummyExecutor)
    monkeypatch.setattr("nodetool.config.settings.get_log_path", lambda f: tmp_path / f)

    agent = Agent(
        name="my agent",
        objective="obj",
        provider=provider,
        model="m",
        task=task,
        output_schema={"type": "string"},
        verbose=False,
    )

    context = ProcessingContext(workspace_dir=str(tmp_path))
    results = []
    async for item in agent.execute(context):
        results.append(item)

    assert agent.get_results() == "final"
    assert subtask.output_schema == json.dumps({"type": "string"})

    # Check that we have the expected types in the results
    result_types = [type(i) for i in results]
    assert TaskUpdate in result_types
    assert SubTaskResult in result_types
    # Check that we have at least the expected number of TaskUpdates
    task_update_count = sum(1 for t in result_types if t == TaskUpdate)
    assert task_update_count >= 3
    # Find the last TaskUpdate in results
    last_task_update = None
    for result in reversed(results):
        if isinstance(result, TaskUpdate):
            last_task_update = result
            break
    assert last_task_update is not None
    assert last_task_update.event == TaskUpdateEvent.TASK_COMPLETED


class DummyTaskExecutor:
    def __init__(self, *, provider, model, processing_context, tools, task, **kwargs):
        self.task = task

    async def execute_tasks(self, context):
        subtask = self.task.subtasks[0]
        yield TaskUpdate(
            task=self.task, subtask=subtask, event=TaskUpdateEvent.SUBTASK_STARTED
        )
        yield Chunk(content="work")
        yield TaskUpdate(
            task=self.task, subtask=subtask, event=TaskUpdateEvent.SUBTASK_COMPLETED
        )
        yield SubTaskResult(subtask=subtask, result="done", is_task_result=True)


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
        inputs,
        output_schema,
        enable_analysis_phase,
        enable_data_contracts_phase,
        use_structured_output,
        verbose,
    ):
        sub = SubTask(content="do")
        task = Task(title="t", subtasks=[sub])
        self.task_plan = TaskPlan(title="p", tasks=[task])

    async def create_task(self, processing_context, objective):
        yield PlanningUpdate(phase="plan", status="ok", content="start")

    async def save_task_plan(self):
        pass


@pytest.mark.asyncio
async def test_agent_execute_with_initial_task(monkeypatch, tmp_path):
    provider = MockProvider([])
    sub = SubTask(content="do")
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
