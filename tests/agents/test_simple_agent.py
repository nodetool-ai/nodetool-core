import json
import asyncio
import pytest

from nodetool.agents.simple_agent import SimpleAgent
from nodetool.chat.providers.base import MockProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import Message, SubTask, Task, ToolCall
from nodetool.workflows.types import Chunk
from nodetool.agents.tools.base import Tool


class DummyTool(Tool):
    name = "dummy"
    description = "a dummy tool"


@pytest.mark.asyncio
async def test_get_execution_tools_info():
    provider = MockProvider([])
    agent = SimpleAgent(
        name="a",
        objective="obj",
        provider=provider,
        model="m",
        tools=[DummyTool()],
        output_type="json",
        output_schema={"type": "string"},
    )
    assert agent._get_execution_tools_info() == "- dummy: a dummy tool"


@pytest.mark.asyncio
async def test_plan_single_subtask_success(tmp_path):
    response = Message(
        role="assistant",
        content=json.dumps({
            "content": "do it",
            "output_type": "json",
            "output_schema": "{\"type\": \"string\"}",
        }),
    )
    provider = MockProvider([response])
    agent = SimpleAgent(
        name="a",
        objective="obj",
        provider=provider,
        model="m",
        tools=[],
        output_type="json",
        output_schema={"type": "string"},
    )
    context = ProcessingContext(workspace_dir=str(tmp_path))
    await agent._plan_single_subtask(context)
    assert agent.task is not None
    assert agent.subtask is not None
    assert agent.subtask.content == "do it"
    assert agent.subtask.output_file == "output.json"


@pytest.mark.asyncio
async def test_plan_single_subtask_bad_json(tmp_path):
    response = Message(role="assistant", content="not json")
    provider = MockProvider([response])
    agent = SimpleAgent(
        name="a",
        objective="obj",
        provider=provider,
        model="m",
        tools=[],
        output_type="json",
        output_schema={"type": "string"},
    )
    context = ProcessingContext(workspace_dir=str(tmp_path))
    with pytest.raises(ValueError):
        await agent._plan_single_subtask(context)


@pytest.mark.asyncio
async def test_execute_yields_results(monkeypatch, tmp_path):
    provider = MockProvider([])
    agent = SimpleAgent(
        name="a",
        objective="obj",
        provider=provider,
        model="m",
        tools=[],
        output_type="json",
        output_schema={"type": "string"},
    )
    async def fake_plan(self, ctx, max_retries=3):
        self.subtask = SubTask(
            content="task",
            output_file="output.json",
            input_files=[],
            output_type="json",
            output_schema="{}",
        )
        self.task = Task(title=self.objective, subtasks=[self.subtask])
    monkeypatch.setattr(SimpleAgent, "_plan_single_subtask", fake_plan)

    class FakeSubTaskContext:
        def __init__(self, *args, **kwargs):
            pass
        async def execute(self):
            yield Chunk(content="hello")
            yield ToolCall(id="1", name="finish_subtask", args={"result": "42"})
    monkeypatch.setattr(
        "nodetool.agents.simple_agent.SubTaskContext", FakeSubTaskContext
    )
    context = ProcessingContext(workspace_dir=str(tmp_path))
    out = []
    async for item in agent.execute(context):
        out.append(item)
    assert any(isinstance(i, Chunk) for i in out)
    assert any(isinstance(i, ToolCall) for i in out)
    assert agent.get_results() == "42"
