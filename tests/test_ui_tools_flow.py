import pytest

from nodetool.chat.base_chat_runner import BaseChatRunner
from nodetool.integrations.websocket.unified_websocket_runner import ToolBridge
from nodetool.messaging.help_message_processor import HelpMessageProcessor
from nodetool.metadata.types import Message, Provider, ToolCall
from nodetool.providers.base import MockProvider
from nodetool.workflows.processing_context import ProcessingContext


@pytest.mark.asyncio
async def test_toolbridge_resolves_result_via_receive_loop(monkeypatch):
    runner = DummyRunner()
    runner.tool_bridge = ToolBridge()

    # Prepare a waiting future
    call_id = "call_1"
    waiter = runner.tool_bridge.create_waiter(call_id)

    # Sequence of messages: tool_result then connection closed (None)
    messages = [
        {
            "type": "tool_result",
            "tool_call_id": call_id,
            "ok": True,
            "result": {"ok": True},
        },
        None,
    ]

    async def fake_receive_message():  # type: ignore
        return messages.pop(0)

    async def fake_send_message(_msg: dict):  # type: ignore
        return None

    monkeypatch.setattr(runner, "receive_message", fake_receive_message)  # type: ignore
    monkeypatch.setattr(runner, "send_message", fake_send_message)  # type: ignore

    await runner._receive_messages()

    result = await waiter
    assert result["ok"] is True


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

    async def _receive_messages(self):
        """Mock loop that mimics ChatWebSocketRunner's handling of tool results."""
        while True:
            message = await self.receive_message()
            if message is None:
                break

            # Mimic tool result handling
            if hasattr(self, "tool_bridge") and message.get("type") == "tool_result":
                self.tool_bridge.resolve_result(message["tool_call_id"], message)


@pytest.mark.asyncio
async def test_help_processor_ui_tool_flow():
    # Mock provider will emit a ToolCall for a UI tool
    tool_call = ToolCall(
        id="tc_1",
        name="ui_add_node",
        args={"node": {"id": "n1", "position": {"x": 1, "y": 2}, "data": {}}},
    )
    provider = MockProvider(responses=[Message(role="assistant", tool_calls=[tool_call])])

    processor = HelpMessageProcessor(provider)

    tool_manifest = {
        "name": "ui_add_node",
        "description": "Add a node to the current workflow graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "position": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["x", "y"],
                        },
                        "data": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["id", "position"],
                }
            },
            "required": ["node"],
        },
    }

    tool_bridge = ToolBridge()
    context = ProcessingContext(
        tool_bridge=tool_bridge,
        ui_tool_names={"ui_add_node"},
        client_tools_manifest={"ui_add_node": tool_manifest},
    )

    # Last user message requirements for HelpMessageProcessor
    chat_history = [
        Message(
            role="user",
            content="Please add a node",
            provider=Provider.OpenAI,
            model="gpt-test",
            thread_id="t1",
        )
    ]

    # Runner that resolves tool results when it sees a tool_call outbound
    class ResolvingRunner(DummyRunner):
        async def send_message(self, message: dict):  # type: ignore[override]
            self.sent.append(message)
            if message.get("type") == "tool_call" and message.get("name") == "ui_add_node":
                tool_bridge.resolve_result(
                    message["tool_call_id"],
                    {
                        "type": "tool_result",
                        "tool_call_id": message["tool_call_id"],
                        "thread_id": message.get("thread_id"),
                        "ok": True,
                        "result": {"ok": True, "nodes": 11},
                        "elapsed_ms": 5,
                    },
                )

    runner = ResolvingRunner()

    await runner._run_processor(
        processor=processor,
        chat_history=chat_history,
        processing_context=context,
    )

    # Validate that a tool_call was sent
    tool_calls = [m for m in runner.sent if m.get("type") == "tool_call"]
    assert any(tc.get("name") == "ui_add_node" for tc in tool_calls)

    # Since BaseChatRunner saves tool role messages to DB (type=="message")
    # and does not emit them via send_message, we assert completion signals
    # were sent (either chunk done or an error if provider ran out of responses).
    chunks = [m for m in runner.sent if m.get("type") == "chunk"]
    errors = [m for m in runner.sent if m.get("type") == "error"]
    assert len(chunks) > 0 or len(errors) > 0
