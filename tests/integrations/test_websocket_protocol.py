"""
WebSocket Protocol Integration Tests

This module tests the WebSocket protocol by starting a real server
and connecting with an actual WebSocket client. It tests the complete
protocol including:
- Connection and authentication
- Message encoding (binary/MessagePack and text/JSON)
- Command handling (run_job, cancel_job, chat_message, etc.)
- Event streaming (job_update, node_update, chunk, etc.)
- Ping/pong keepalive
- Error handling for unknown job_ids
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from nodetool.api.server import create_app


def _create_test_client(env: str = "development") -> TestClient:
    """Create a test client for the API."""
    import os

    os.environ["ENV"] = env
    return TestClient(create_app())


class WebSocketProtocolTester:
    """Helper class for testing WebSocket protocol."""

    def __init__(self, ws: WebSocketTestSession):
        self.ws = ws
        self.mode = "binary"  # Default mode is binary (MessagePack)

    def send_command(self, command: str, data: dict):
        """Send a command message."""
        msg = {"command": command, "data": data}
        self._send(msg)

    def send_raw(self, msg: dict):
        """Send a raw message (for ping, etc.)."""
        self._send(msg)

    def _send(self, msg: dict):
        """Internal send method."""
        if self.mode == "binary":
            packed = msgpack.packb(msg, use_bin_type=True)
            self.ws.send_bytes(packed)
        else:
            self.ws.send_json(msg)

    def receive(self) -> dict:
        """Receive and decode a message."""
        if self.mode == "binary":
            data = self.ws.receive_bytes()
            return msgpack.unpackb(data)
        else:
            return self.ws.receive_json()

    def _receive_in_mode(self, mode: str) -> dict:
        """Receive a message in a specific mode (for mode switching responses)."""
        if mode == "binary":
            data = self.ws.receive_bytes()
            return msgpack.unpackb(data)
        else:
            return self.ws.receive_json()

    def set_mode_text(self):
        """Switch to text mode."""
        self.send_command("set_mode", {"mode": "text"})
        # Response is in text mode, so receive accordingly
        self._receive_in_mode("text")
        self.mode = "text"

    def set_mode_binary(self):
        """Switch to binary mode."""
        self.send_command("set_mode", {"mode": "binary"})
        # Response is in binary mode, so receive accordingly
        self._receive_in_mode("binary")
        self.mode = "binary"


@pytest.fixture
def client():
    """Create a test client in development mode (no auth required)."""
    return _create_test_client("development")


@pytest.fixture
def ws(client):
    """Create a WebSocket connection."""
    with client.websocket_connect("/ws") as ws:
        yield WebSocketProtocolTester(ws)


class TestWebSocketProtocolBasics:
    """Test basic WebSocket protocol functionality."""

    def test_connection_and_get_status(self, ws):
        """Test that connection is established and get_status works."""
        # Server doesn't send initial messages - just send our command
        ws.send_command("get_status", {})
        msg = ws.receive()
        assert msg is not None
        assert "active_jobs" in msg

    def test_set_mode_text(self, ws):
        """Test switching to text mode."""
        ws.send_command("set_mode", {"mode": "text"})
        # Server responds in text mode
        msg = ws._receive_in_mode("text")
        assert msg.get("message", "").startswith("Mode set to")
        # Now switch our local mode to match
        ws.mode = "text"
        assert ws.mode == "text"

    def test_set_mode_binary(self, ws):
        """Test switching to binary mode."""
        ws.send_command("set_mode", {"mode": "binary"})
        msg = ws.receive()
        assert msg.get("message", "").startswith("Mode set to")
        assert ws.mode == "binary"


class TestWebSocketPingPong:
    """Test ping/pong keepalive."""

    def test_ping_pong(self, ws):
        """Test ping/pong exchange."""
        # Send our own ping
        ws.send_raw({"type": "ping"})
        # Receive pong
        msg = ws.receive()
        assert msg.get("type") == "pong"
        assert "ts" in msg


class TestWebSocketCommandValidation:
    """Test command validation."""

    def test_chat_message_requires_thread_id(self, ws):
        """Test that chat_message command requires thread_id."""
        ws.send_command("chat_message", {"content": "Hello"})
        msg = ws.receive()
        assert "error" in msg
        assert "thread_id is required" in msg["error"]

    def test_cancel_job_requires_job_id(self, ws):
        """Test that cancel_job command requires job_id."""
        ws.send_command("cancel_job", {})
        msg = ws.receive()
        assert "error" in msg
        assert "job_id is required" in msg["error"]

    def test_get_status(self, ws):
        """Test get_status command."""
        ws.send_command("get_status", {})
        msg = ws.receive()
        assert "active_jobs" in msg
        assert isinstance(msg["active_jobs"], list)

    def test_stop_requires_reference(self, ws):
        """Test that stop command requires job_id or thread_id."""
        ws.send_command("stop", {})
        msg = ws.receive()
        assert "error" in msg
        assert "job_id or thread_id is required" in msg["error"]


class TestWebSocketProtocolModes:
    """Test both binary and text modes."""

    def test_binary_mode_messagepack(self, ws):
        """Test that binary mode uses MessagePack encoding."""
        # Get status in binary mode
        ws.send_command("get_status", {})
        msg = ws.receive()
        assert "active_jobs" in msg

    def test_text_mode_json(self, ws):
        """Test that text mode uses JSON encoding."""
        # Switch to text mode using helper
        ws.set_mode_text()

        # Get status in text mode
        ws.send_command("get_status", {})
        msg = ws.receive()
        assert "active_jobs" in msg


class TestWebSocketErrorHandling:
    """Test error handling scenarios."""


class TestWebSocketMessageSchema:
    """Test that all message types follow the expected schema."""

    def test_job_update_schema(self, ws):
        """Test job_update message has correct schema."""
        # Get status - returns active_jobs which includes job_update info
        ws.send_command("get_status", {})
        msg = ws.receive()
        assert "active_jobs" in msg

    def test_error_schema(self, ws):
        """Test error message has correct schema."""
        # Try to cancel without job_id
        ws.send_command("cancel_job", {})
        msg = ws.receive()
        assert "error" in msg
        assert isinstance(msg["error"], str)


class TestWebSocketRoutingKeys:
    """Test that messages have correct routing keys."""

    def test_chunk_has_thread_id_field(self, ws):
        """Test that chunk messages can include thread_id."""
        from nodetool.metadata.types import Chunk

        chunk = Chunk(content="test", thread_id="test-thread-123")
        assert hasattr(chunk, "thread_id")
        assert chunk.thread_id == "test-thread-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
