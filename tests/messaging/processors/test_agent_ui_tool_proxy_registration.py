from unittest.mock import MagicMock

import pytest

from nodetool.messaging.agent_message_processor import AgentMessageProcessor
from nodetool.metadata.types import Message, Provider
from nodetool.providers.base import BaseProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


@pytest.mark.asyncio
async def test_agent_processor_registers_ui_tool_proxies(monkeypatch):
    created_agents: list[object] = []

    class DummyAgent:
        def __init__(self, *args, tools=None, **kwargs):
            self.tools = tools or []
            created_agents.append(self)

        async def execute(self, _context):
            yield Chunk(content="", done=True)

    monkeypatch.setattr("nodetool.agents.agent.Agent", DummyAgent)

    provider = MagicMock(spec=BaseProvider)
    processor = AgentMessageProcessor(provider=provider)

    tool_manifest = {
        "name": "ui_add_node",
        "description": "Add a node to the current workflow graph.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    context = ProcessingContext(
        tool_bridge=object(),
        client_tools_manifest={"ui_add_node": tool_manifest},
    )

    chat_history = [
        Message(
            role="user",
            instructions="Add a node",
            provider=Provider.OpenAI,
            model="gpt-test",
        )
    ]

    await processor.process(chat_history=chat_history, processing_context=context)

    assert created_agents, "Agent should be instantiated"
    agent_tools = getattr(created_agents[0], "tools", [])
    assert any(getattr(t, "name", None) == "ui_add_node" for t in agent_tools)
