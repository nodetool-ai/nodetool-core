import msgpack
import pytest
from fastapi.testclient import TestClient
from nodetool.metadata.types import Message
from nodetool.chat.providers.base import MockProvider
from nodetool.common import chat_websocket_runner


def test_chat_websocket_basic(monkeypatch, client: TestClient):
    async def fake_provider_from_model(model: str):
        return MockProvider([Message(role="assistant", content="Hi")])

    monkeypatch.setattr(chat_websocket_runner, "provider_from_model", fake_provider_from_model)

    with client.websocket_connect("/chat") as ws:
        user_msg = Message(role="user", content="Hello", model="test-model")
        ws.send_bytes(msgpack.packb(user_msg.model_dump()))
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"
        assert data["content"] == "Hi"
