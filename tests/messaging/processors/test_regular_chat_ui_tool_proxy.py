"""Tests for UI tool proxy registration in RegularChatProcessor."""

from unittest.mock import MagicMock

import pytest

from nodetool.messaging.regular_chat_processor import RegularChatProcessor
from nodetool.metadata.types import Message, Provider
from nodetool.providers.base import BaseProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


@pytest.mark.asyncio
async def test_regular_chat_processor_registers_ui_tool_proxies(monkeypatch):
    """Test that RegularChatProcessor adds UI tool proxies when tool_bridge is available."""
    # Track tools passed to generate_messages
    captured_tools = []

    async def mock_generate_messages(**kwargs):
        captured_tools.extend(kwargs.get("tools", []))
        # Return a simple chunk to end processing
        yield Chunk(content="response", done=True)

    provider = MagicMock(spec=BaseProvider)
    provider.generate_messages = mock_generate_messages
    provider.cost = 0.0
    provider.log_provider_call = MagicMock(return_value=None)

    processor = RegularChatProcessor(provider=provider)

    tool_manifest = {
        "name": "ui_add_node",
        "description": "Add a node to the current workflow graph.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    context = ProcessingContext(
        tool_bridge=object(),
        ui_tool_names={"ui_add_node"},
        client_tools_manifest={"ui_add_node": tool_manifest},
    )

    chat_history = [
        Message(
            role="user",
            content="Add a node",
            provider=Provider.OpenAI,
            model="gpt-test",
        )
    ]

    await processor.process(chat_history=chat_history, processing_context=context)

    # Check that UI tool proxy was included in tools
    assert any(getattr(t, "name", None) == "ui_add_node" for t in captured_tools), (
        "UI tool proxy should be registered when client_tools_manifest is provided"
    )
