import pytest

from nodetool.chat.base_chat_runner import BaseChatRunner
from nodetool.chat.chat_websocket_runner import ChatWebSocketRunner
from nodetool.messaging.help_message_processor import HelpMessageProcessor
from nodetool.metadata.types import Message, Provider, ToolCall
from nodetool.providers.base import MockProvider
from nodetool.workflows.processing_context import ProcessingContext


class DummyRunner(BaseChatRunner):
    def __init__(self):
        super().__init__(auth_token=None)
        self.sent: list[dict] = []

    async def connect(self, **kwargs):  # type: ignore[override]
        return None

    async def disconnect(self):  # type: ignore[override]
        return None

    async def send_message(self, message: dict):  # type: ignore[override]
        self.sent.append(message)

    async def receive_message(self):  # type: ignore[override]
        return None


class ResolvingRunner(DummyRunner):
    async def send_message(self, message: dict):  # type: ignore[override]
        self.sent.append(message)
        if message.get("type") == "tool_call":
            call_id = message["tool_call_id"]
            tool_bridge.resolve_result(
                call_id,
                {
                    "type": "tool_result",
                    "tool_call_id": call_id,
                    "thread_id": message.get("thread_id"),
                    "ok": True,
                    "result": {"ok": True},
                    "elapsed_ms": 1,
                },
            )


@pytest.mark.asyncio
async def test_help_processor_appends_tool_history():
    tool_call1 = ToolCall(id="tc1", name="ui_add_node", args={"node": {}})
    tool_call2 = ToolCall(id="tc2", name="ui_add_node", args={"node": {}})
    provider = MockProvider(
        responses=[
            Message(role="assistant", tool_calls=[tool_call1]),
            Message(role="assistant", tool_calls=[tool_call2]),
            Message(role="assistant", content="done"),
        ]
    )
    processor = HelpMessageProcessor(provider)

    tool_manifest = {
        "name": "ui_add_node",
        "description": "Add a node to the current workflow graph.",
        "parameters": {
            "type": "object",
            "properties": {"node": {"type": "object"}},
            "required": ["node"],
        },
    }

    global tool_bridge
    tool_bridge = ChatWebSocketRunner().tool_bridge
    context = ProcessingContext(
        tool_bridge=tool_bridge,
        ui_tool_names={"ui_add_node"},
        client_tools_manifest={"ui_add_node": tool_manifest},
    )

    chat_history = [
        Message(
            role="user",
            content="Please add nodes",
            provider=Provider.OpenAI,
            model="gpt-test",
            thread_id="t1",
        )
    ]

    runner = ResolvingRunner()

    await runner._run_processor(
        processor=processor,
        chat_history=chat_history,
        processing_context=context,
    )

    assert len(provider.call_log) == 3
    third_call_msgs = provider.call_log[2]["messages"]
    tool_ids = {m.tool_call_id for m in third_call_msgs if m.role == "tool"}
    assert tool_ids == {"tc1", "tc2"}
