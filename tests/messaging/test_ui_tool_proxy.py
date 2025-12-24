import asyncio

import pytest


class FakeToolBridge:
    def __init__(self):
        self._futures = {}

    def create_waiter(self, tool_call_id: str) -> asyncio.Future:
        fut = asyncio.get_running_loop().create_future()
        self._futures[tool_call_id] = fut
        return fut

    def resolve(self, tool_call_id: str, payload: dict):
        fut = self._futures.pop(tool_call_id, None)
        assert fut is not None
        fut.set_result(payload)


class FakeContext:
    def __init__(self, tool_bridge: FakeToolBridge):
        self.tool_bridge = tool_bridge
        self.sent = []
        self.thread_id = "t1"

    async def send_message(self, msg: dict):
        self.sent.append(msg)


@pytest.mark.asyncio
async def test_ui_tool_proxy_error_payload_returns_error_dict(monkeypatch):
    from nodetool.messaging.help_message_processor import UIToolProxy

    tool = UIToolProxy(
        {
            "name": "ui_test_tool",
            "description": "test",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    )

    bridge = FakeToolBridge()
    ctx = FakeContext(bridge)

    monkeypatch.setattr("uuid.uuid4", lambda: "call_1")

    task = asyncio.create_task(tool.process(ctx, params={"x": 1}))

    # ensure tool_call was sent before resolving
    await asyncio.sleep(0)
    assert ctx.sent and ctx.sent[0]["type"] == "tool_call"
    assert ctx.sent[0]["tool_call_id"] == "call_1"

    bridge.resolve("call_1", {"ok": False, "error": "boom"})
    result = await task

    assert result == {"error": "Frontend tool execution failed: boom"}
