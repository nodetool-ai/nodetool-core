import json
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
        output_schema={"type": "string"},
        inputs={},
    )
    assert agent._get_execution_tools_info() == "- dummy: a dummy tool"


@pytest.mark.asyncio
async def test_execute_yields_results(monkeypatch, tmp_path):
    provider = MockProvider([
        Message(
            content="I will complete this task by providing the result.",
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="finish_1",
                    name="finish_subtask",
                    args={"result": "42"}
                )
            ]
        )
    ])
    agent = SimpleAgent(
        name="a",
        objective="obj",
        provider=provider,
        model="m",
        tools=[],
        output_schema={"type": "string"},
        inputs={},
    )

    context = ProcessingContext()
    async for _ in agent.execute(context):
        pass
    print(agent.get_results())
    assert agent.get_results() == "42"
