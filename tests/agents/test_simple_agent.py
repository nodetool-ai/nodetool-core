import pytest

from nodetool.agents.simple_agent import SimpleAgent
from nodetool.providers.base import MockProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import Message
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
    provider = MockProvider(
        [
            Message(
                content='```json\n{"status": "completed", "result": "42"}\n```',
                role="assistant",
            )
        ]
    )
    agent = SimpleAgent(
        name="a",
        objective="obj",
        provider=provider,
        model="m",
        tools=[],
        output_schema={"type": "string"},
        inputs={},
    )

    context = ProcessingContext(workspace_dir=str(tmp_path))
    async for _ in agent.execute(context):
        pass
    assert agent.get_results() == "42"
