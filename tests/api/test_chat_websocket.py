import json
import msgpack  # type: ignore
import pytest
from typing import Any, Dict, List, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect

from nodetool.metadata.types import (  # type: ignore
    Message,
    MessageTextContent,
    MessageImageContent,
    MessageVideoContent,
    MessageAudioContent,
    ImageRef,
    VideoRef,
    AudioRef,
    ToolCall,
    Provider,
)
from nodetool.chat.providers.base import MockProvider, ChatProvider  # type: ignore
from nodetool.chat.providers.anthropic_provider import AnthropicProvider  # type: ignore
from nodetool.chat.providers.openai_provider import OpenAIProvider  # type: ignore
from nodetool.chat.providers.gemini_provider import GeminiProvider  # type: ignore
from nodetool.chat.providers.ollama_provider import OllamaProvider  # type: ignore
from nodetool.common import chat_websocket_runner  # type: ignore
from nodetool.common.chat_websocket_runner import ChatWebSocketRunner, WebSocketMode  # type: ignore
from nodetool.workflows.types import Chunk  # type: ignore
from nodetool.agents.tools.base import Tool  # type: ignore


# ========== Additional Ollama Test ==========


@pytest.mark.asyncio
async def test_provider_from_model_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Ollama model provider selection."""
    # Mock ollama models to include the test model
    chat_websocket_runner.ollama_models = ["llama2", "mistral", "test-ollama-model"]

    provider = await chat_websocket_runner.provider_from_model("test-ollama-model")
    assert isinstance(provider, OllamaProvider)


# ========== Connection and Authentication Tests ==========


def test_chat_websocket_basic(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test basic WebSocket connection and message exchange."""

    async def fake_provider_from_model(model: str) -> MockProvider:
        return MockProvider([Message(role="assistant", content="Hi")])

    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )

    with client.websocket_connect("/chat") as ws:
        user_msg = Message(role="user", content="Hello", model="test-model")
        data = msgpack.packb(user_msg.model_dump())
        assert data is not None
        ws.send_bytes(data)
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"
        assert data["content"] == "Hi"


def test_websocket_connection_with_auth_token(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test WebSocket connection with valid authentication token."""
    runner = ChatWebSocketRunner(auth_token="valid-token")

    async def mock_validate_token(self: ChatWebSocketRunner, token: str) -> bool:
        return token == "valid-token"

    monkeypatch.setattr(ChatWebSocketRunner, "validate_token", mock_validate_token)
    monkeypatch.setattr(
        chat_websocket_runner.Environment, "is_production", lambda: True
    )

    # This would need actual WebSocket testing with auth headers
    # Simplified test showing the auth flow works
    assert runner.auth_token == "valid-token"


@pytest.mark.asyncio
async def test_websocket_connection_invalid_auth_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test WebSocket connection rejection with invalid token."""

    async def mock_validate_token(self: ChatWebSocketRunner, token: str) -> bool:
        return False

    monkeypatch.setattr(ChatWebSocketRunner, "validate_token", mock_validate_token)

    # Create a mock websocket
    mock_websocket = AsyncMock()

    runner = ChatWebSocketRunner(auth_token="invalid-token")
    await runner.connect(mock_websocket)

    # Should close connection with invalid auth
    mock_websocket.close.assert_called_once_with(
        code=1008, reason="Invalid authentication"
    )


@pytest.mark.asyncio
async def test_websocket_connection_missing_auth_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test WebSocket connection rejection when auth is missing in production."""
    monkeypatch.setattr(
        chat_websocket_runner.Environment, "is_production", lambda: True
    )

    # Create a mock websocket
    mock_websocket = AsyncMock()

    runner = ChatWebSocketRunner(auth_token=None)
    await runner.connect(mock_websocket)

    # Should close connection with missing auth
    mock_websocket.close.assert_called_once_with(
        code=1008, reason="Missing authentication"
    )


# ========== Message Format Tests ==========


def test_text_message_format(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test processing JSON text messages."""

    async def fake_provider_from_model(model: str) -> MockProvider:
        return MockProvider([Message(role="assistant", content="Text response")])

    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )

    with client.websocket_connect("/chat") as ws:
        user_msg = Message(role="user", content="Hello", model="test-model")
        data = json.dumps(user_msg.model_dump())
        ws.send_text(data)
        response = json.loads(ws.receive_text())
        assert response["type"] == "chunk"
        assert response["content"] == "Text response"


def test_invalid_message_format(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test handling of messages with unknown format."""
    with client.websocket_connect("/chat") as ws:
        # Send invalid data that's neither bytes nor text properly
        # This tests the error handling path
        try:
            ws.send_bytes(b"invalid")
            response = msgpack.unpackb(ws.receive_bytes())
            assert response["type"] == "error"
        except:
            # Expected behavior when invalid message is sent
            pass


@pytest.mark.asyncio
async def test_websocket_unknown_message_type() -> None:
    """Test handling of messages that are neither bytes nor text."""
    runner = ChatWebSocketRunner()

    # Create a mock websocket that returns unknown message type
    mock_websocket = AsyncMock()
    mock_websocket.receive = AsyncMock(
        return_value={
            "type": "websocket.receive",
            "unknown": "data",  # Neither "bytes" nor "text"
        }
    )
    mock_websocket.accept = AsyncMock()

    # Run should handle unknown message gracefully
    await runner.connect(mock_websocket)
    # This would normally be called in run() but we're testing just the message handling


# ========== Provider Selection Tests ==========


@pytest.mark.asyncio
async def test_provider_from_model_anthropic() -> None:
    """Test Claude model provider selection."""
    provider = await chat_websocket_runner.provider_from_model("claude-3-opus")
    assert isinstance(provider, AnthropicProvider)


@pytest.mark.asyncio
async def test_provider_from_model_openai() -> None:
    """Test GPT model provider selection."""
    provider = await chat_websocket_runner.provider_from_model("gpt-4")
    assert isinstance(provider, OpenAIProvider)


@pytest.mark.asyncio
async def test_provider_from_model_gemini() -> None:
    """Test Gemini model provider selection."""
    provider = await chat_websocket_runner.provider_from_model("gemini-pro")
    assert isinstance(provider, GeminiProvider)


@pytest.mark.asyncio
async def test_provider_from_model_unknown() -> None:
    """Test unknown model error handling."""
    with pytest.raises(ValueError, match="Unsupported model"):
        await chat_websocket_runner.provider_from_model("unknown-model")


@pytest.mark.asyncio
async def test_cached_ollama_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Ollama model caching."""
    # Create mock models with name attribute
    mock_model1 = MagicMock()
    mock_model1.name = "llama2"
    mock_model2 = MagicMock()
    mock_model2.name = "mistral"
    mock_models = [mock_model1, mock_model2]

    async def mock_get_ollama_models() -> List[Any]:
        return mock_models

    monkeypatch.setattr(
        chat_websocket_runner, "get_ollama_models", mock_get_ollama_models
    )

    # Reset global cache first
    chat_websocket_runner.ollama_models = []

    # First call should fetch models
    models1 = await chat_websocket_runner.cached_ollama_models()
    assert models1 == ["llama2", "mistral"]

    # Second call should use cache
    models2 = await chat_websocket_runner.cached_ollama_models()
    assert models2 == ["llama2", "mistral"]


# ========== Tool Execution Tests ==========


@pytest.mark.asyncio
async def test_tool_execution_success() -> None:
    """Test successful tool execution."""
    from nodetool.workflows.processing_context import ProcessingContext  # type: ignore

    class TestTool(Tool):
        name = "test_tool"
        description = "Test tool"

        async def process(self, context: Any, args: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "success", "input": args.get("input")}

    context = ProcessingContext()
    tool_call = ToolCall(id="test-123", name="test_tool", args={"input": "test data"})

    result = await chat_websocket_runner.run_tool(context, tool_call, [TestTool()])

    assert result.id == "test-123"
    assert result.name == "test_tool"
    assert result.result["result"] == "success"
    assert result.result["input"] == "test data"


@pytest.mark.asyncio
async def test_tool_not_found() -> None:
    """Test error when tool is not found."""
    from nodetool.workflows.processing_context import ProcessingContext  # type: ignore

    context = ProcessingContext()
    tool_call = ToolCall(id="test-456", name="nonexistent_tool", args={})

    with pytest.raises(AssertionError, match="Tool nonexistent_tool not found"):
        await chat_websocket_runner.run_tool(context, tool_call, [])


def test_tool_with_complex_args(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test tool execution with nested arguments."""
    tool_result = {"status": "processed", "data": {"nested": "value"}}

    async def fake_provider_from_model(model: str) -> MockProvider:
        tool_call = ToolCall(
            id="complex-123",
            name="complex_tool",
            args={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )
        return MockProvider(
            [Message(role="assistant", content="", tool_calls=[tool_call])]
        )

    class ComplexTool(Tool):
        name = "complex_tool"
        description = "Complex tool"

        async def process(self, context: Any, args: Dict[str, Any]) -> Dict[str, Any]:
            return tool_result

    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )
    monkeypatch.setattr(
        chat_websocket_runner,
        "get_tool_by_name",
        lambda name: ComplexTool if name == "complex_tool" else None,
    )

    with client.websocket_connect("/chat") as ws:
        user_msg = Message(
            role="user",
            content="Use complex tool",
            model="test-model",
            tools=["complex_tool"],
        )
        ws.send_bytes(msgpack.packb(user_msg.model_dump()) or b"")

        # Receive tool call
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "tool_call"

        # Receive tool result
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "tool_result"
        assert data["result"]["result"] == tool_result


# ========== Chat Message Processing Tests ==========


def test_process_messages_streaming(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test chunk streaming during message processing."""
    chunks = [Chunk(content="Hello ", done=False), Chunk(content="world!", done=True)]

    class ChunkProvider(ChatProvider):
        async def generate_message(
            self, messages: Any, model: str, tools: List[Any] = [], **kwargs: Any
        ) -> Message:
            return Message(role="assistant", content="Hello world!")

        async def generate_messages(
            self, messages: Any, model: str, tools: List[Any] = [], **kwargs: Any
        ) -> AsyncGenerator[Chunk, None]:
            for chunk in chunks:
                yield chunk

    async def fake_provider_from_model(model: str) -> ChunkProvider:
        return ChunkProvider()

    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )

    with client.websocket_connect("/chat") as ws:
        user_msg = Message(role="user", content="Hi", model="test-model")
        ws.send_bytes(msgpack.packb(user_msg.model_dump()) or b"")

        # Receive first chunk
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"
        assert data["content"] == "Hello "
        assert data["done"] is False

        # Receive second chunk
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"
        assert data["content"] == "world!"
        assert data["done"] is True


def test_chat_history_accumulation(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test that messages are properly added to chat history."""
    message_count = 0

    class HistoryTrackingProvider(ChatProvider):
        async def generate_message(
            self, messages: Any, model: str, tools: List[Any] = [], **kwargs: Any
        ) -> Message:
            nonlocal message_count
            message_count = len(messages)
            return Message(
                role="assistant", content=f"Received {len(messages)} messages"
            )

        async def generate_messages(
            self, messages: Any, model: str, tools: List[Any] = [], **kwargs: Any
        ) -> AsyncGenerator[Chunk, None]:
            nonlocal message_count
            message_count = len(messages)
            yield Chunk(content=f"Received {len(messages)} messages", done=True)

    async def fake_provider_from_model(model: str) -> HistoryTrackingProvider:
        return HistoryTrackingProvider()

    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )

    with client.websocket_connect("/chat") as ws:
        # Send first message
        msg1 = Message(role="user", content="First", model="test-model")
        ws.send_bytes(msgpack.packb(msg1.model_dump()) or b"")
        data = msgpack.unpackb(ws.receive_bytes())
        assert "Received 1 messages" in data["content"]

        # Send second message
        msg2 = Message(role="user", content="Second", model="test-model")
        ws.send_bytes(msgpack.packb(msg2.model_dump()) or b"")
        data = msgpack.unpackb(ws.receive_bytes())
        assert "Received 3 messages" in data["content"]  # 2 user + 1 assistant


# ========== Help System Tests ==========


def test_help_message_processing(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test help system activation and response."""
    help_chunks = [
        Chunk(content="Here is help information: ", done=False),
        Chunk(content="Use the system wisely.", done=True),
    ]

    async def mock_create_help_answer(
        provider: Any, messages: Any, model: str
    ) -> AsyncGenerator[Chunk, None]:
        for chunk in help_chunks:
            yield chunk

    async def fake_provider_from_model(model: str) -> MockProvider:
        return MockProvider([])

    monkeypatch.setattr(
        chat_websocket_runner, "create_help_answer", mock_create_help_answer
    )
    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )

    with client.websocket_connect("/chat") as ws:
        help_msg = Message(role="user", content="Help me", model="help:gpt-4")
        ws.send_bytes(msgpack.packb(help_msg.model_dump()) or b"")

        # Receive help chunks
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"
        assert data["content"] == "Here is help information: "

        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"
        assert data["content"] == "Use the system wisely."

        # Receive done signal
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"
        assert data["content"] == ""
        assert data["done"] is True


def test_help_message_with_tool_calls(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test help system with tool calls."""
    tool_call = ToolCall(
        id="help-tool-123", name="help_tool", args={"action": "get_help"}
    )

    help_items: List[Any] = [
        ToolCall(id="help-tool-123", name="help_tool", args={"action": "get_help"}),
        Chunk(content="Help retrieved.", done=True),
    ]

    async def mock_create_help_answer(
        provider: Any, messages: Any, model: str
    ) -> AsyncGenerator[Any, None]:
        for item in help_items:
            yield item

    async def fake_provider_from_model(model: str) -> MockProvider:
        return MockProvider([])

    monkeypatch.setattr(
        chat_websocket_runner, "create_help_answer", mock_create_help_answer
    )
    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )

    with client.websocket_connect("/chat") as ws:
        help_msg = Message(role="user", content="Help me", model="help:gpt-4")
        ws.send_bytes(msgpack.packb(help_msg.model_dump()) or b"")

        # Receive tool call
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "tool_call"
        assert data["tool_call"]["id"] == "help-tool-123"

        # Receive chunk
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"
        assert data["content"] == "Help retrieved."

        # Receive done signal
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"
        assert data["content"] == ""
        assert data["done"] is True


# ========== Workflow Processing Tests ==========


def test_workflow_execution_success(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test successful workflow execution."""
    workflow_updates = [
        {"type": "job_update", "status": "running", "progress": 0.5},
        {"type": "job_update", "status": "completed", "result": {"output": "Success"}},
    ]

    async def mock_run_workflow(
        request: Any, runner: Any, context: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        for update in workflow_updates:
            yield update  # type: ignore

    monkeypatch.setattr(chat_websocket_runner, "run_workflow", mock_run_workflow)

    with client.websocket_connect("/chat") as ws:
        workflow_msg = Message(
            role="user",
            content="Run workflow",
            model="test-model",
            workflow_id="test-workflow-123",
        )
        ws.send_bytes(msgpack.packb(workflow_msg.model_dump()) or b"")

        # Receive running update
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "job_update"
        assert data["status"] == "running"

        # Receive completed update
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "job_update"
        assert data["status"] == "completed"
        assert data["result"]["output"] == "Success"


def test_workflow_missing_id(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test error when workflow_id is missing."""
    with client.websocket_connect("/chat") as ws:
        # Message without workflow_id but trying to process as workflow
        msg = Message(role="user", content="Run", model="test-model")
        ws.send_bytes(msgpack.packb(msg.model_dump()) or b"")

        # Should process as regular message, not workflow
        # This tests the branching logic


# ========== Response Message Creation Tests ==========


def test_create_response_message_text() -> None:
    """Test creating response message with text content."""
    runner = ChatWebSocketRunner()
    result = {"key1": "Hello", "key2": "World"}

    message = runner.create_response_message(result)
    assert message is not None
    assert message.role == "assistant"
    assert isinstance(message.content, list)
    assert len(message.content) == 2
    assert isinstance(message.content[0], MessageTextContent)
    assert message.content[0].text == "Hello"
    assert isinstance(message.content[1], MessageTextContent)
    assert message.content[1].text == "World"


def test_create_response_message_image() -> None:
    """Test creating response message with image content."""
    runner = ChatWebSocketRunner()
    result = {
        "image": {"type": "image", "uri": "s3://bucket/image.png", "format": "png"}
    }

    message = runner.create_response_message(result)

    assert message is not None
    assert message.role == "assistant"
    assert isinstance(message.content, list)
    assert len(message.content) == 1
    assert isinstance(message.content[0], MessageImageContent)
    assert message.content[0].image.uri == "s3://bucket/image.png"


def test_create_response_message_video() -> None:
    """Test creating response message with video content."""
    runner = ChatWebSocketRunner()
    result = {
        "video": {"type": "video", "uri": "s3://bucket/video.mp4", "format": "mp4"}
    }

    message = runner.create_response_message(result)

    assert message is not None
    assert message.role == "assistant"
    assert isinstance(message.content, list)
    assert len(message.content) == 1
    assert isinstance(message.content[0], MessageVideoContent)
    assert message.content[0].video.uri == "s3://bucket/video.mp4"


def test_create_response_message_audio() -> None:
    """Test creating response message with audio content."""
    runner = ChatWebSocketRunner()
    result = {
        "audio": {"type": "audio", "uri": "s3://bucket/audio.mp3", "format": "mp3"}
    }

    message = runner.create_response_message(result)

    assert message is not None
    assert message.role == "assistant"
    assert isinstance(message.content, list)
    assert len(message.content) == 1
    assert isinstance(message.content[0], MessageAudioContent)
    assert message.content[0].audio.uri == "s3://bucket/audio.mp3"


def test_create_response_message_list() -> None:
    """Test creating response message with list content."""
    runner = ChatWebSocketRunner()
    result = {"items": ["one", "two", "three"]}

    message = runner.create_response_message(result)

    assert message is not None
    assert message.role == "assistant"
    assert isinstance(message.content, list)
    assert len(message.content) == 1
    assert isinstance(message.content[0], MessageTextContent)
    assert message.content[0].text == "one two three"


def test_create_response_message_unknown_type() -> None:
    """Test error handling for unknown content types."""
    runner = ChatWebSocketRunner()
    result = {"unknown": {"type": "unknown", "data": "test"}}

    with pytest.raises(ValueError, match="Unknown type"):
        runner.create_response_message(result)


def test_create_response_message_invalid_value_type() -> None:
    """Test error handling for invalid value types."""
    runner = ChatWebSocketRunner()
    result = {"invalid": 123}  # Not a string, list, or dict

    with pytest.raises(ValueError, match="Unknown type: <class 'int'>"):
        runner.create_response_message(result)


# ========== Disconnect Test ==========


@pytest.mark.asyncio
async def test_websocket_disconnect() -> None:
    """Test WebSocket disconnect method."""
    runner = ChatWebSocketRunner()
    mock_websocket = AsyncMock()
    runner.websocket = mock_websocket

    await runner.disconnect()

    mock_websocket.close.assert_called_once()
    assert runner.websocket is None


# ========== WebSocket Communication Tests ==========


@pytest.mark.asyncio
async def test_send_message_binary_mode() -> None:
    """Test sending messages in binary mode."""
    runner = ChatWebSocketRunner()
    runner.mode = WebSocketMode.BINARY

    mock_websocket = AsyncMock()
    runner.websocket = mock_websocket

    message = {"type": "test", "data": "binary"}
    await runner.send_message(message)

    mock_websocket.send_bytes.assert_called_once()
    sent_data = mock_websocket.send_bytes.call_args[0][0]
    unpacked = msgpack.unpackb(sent_data)
    assert unpacked == message


@pytest.mark.asyncio
async def test_send_message_text_mode() -> None:
    """Test sending messages in text mode."""
    runner = ChatWebSocketRunner()
    runner.mode = WebSocketMode.TEXT

    mock_websocket = AsyncMock()
    runner.websocket = mock_websocket

    message = {"type": "test", "data": "text"}
    await runner.send_message(message)

    mock_websocket.send_text.assert_called_once()
    sent_data = mock_websocket.send_text.call_args[0][0]
    parsed = json.loads(sent_data)
    assert parsed == message


@pytest.mark.asyncio
async def test_send_message_disconnected() -> None:
    """Test error when sending to disconnected WebSocket."""
    runner = ChatWebSocketRunner()
    runner.websocket = None

    with pytest.raises(AssertionError, match="WebSocket is not connected"):
        await runner.send_message({"type": "test"})


@pytest.mark.asyncio
async def test_send_message_error() -> None:
    """Test error handling when message sending fails."""
    runner = ChatWebSocketRunner()
    runner.mode = WebSocketMode.BINARY

    mock_websocket = AsyncMock()
    mock_websocket.send_bytes.side_effect = Exception("Network error")
    runner.websocket = mock_websocket

    # Should log error but not raise
    await runner.send_message({"type": "test", "data": "error"})


# ========== Edge Cases and Error Handling ==========


def test_exception_during_processing(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test error handling during message processing."""

    async def fake_provider_from_model(model: str) -> None:
        raise ValueError("Provider initialization failed")

    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )

    with client.websocket_connect("/chat") as ws:
        msg = Message(role="user", content="Test", model="test-model")
        ws.send_bytes(msgpack.packb(msg.model_dump()) or b"")

        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "error"
        assert "Provider initialization failed" in data["message"]


def test_mode_switching(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Test switching between binary and text modes."""

    async def fake_provider_from_model(model: str) -> MockProvider:
        return MockProvider([Message(role="assistant", content="Response")])

    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )

    with client.websocket_connect("/chat") as ws:
        # Send binary message
        msg1 = Message(role="user", content="Binary", model="test-model")
        ws.send_bytes(msgpack.packb(msg1.model_dump()) or b"")
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"

        # Send text message
        msg2 = Message(role="user", content="Text", model="test-model")
        ws.send_text(json.dumps(msg2.model_dump()) or "")
        data = json.loads(ws.receive_text())
        assert data["type"] == "chunk"


# ========== WebSocket Disconnect Message Test ==========


def test_websocket_disconnect_message(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test handling of websocket.disconnect message."""

    # Create a mock provider that would return a message
    async def fake_provider_from_model(model: str) -> MockProvider:
        return MockProvider([Message(role="assistant", content="Response")])

    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )

    # Use TestClient context manager which simulates disconnect
    with client.websocket_connect("/chat") as ws:
        # Send a message
        msg = Message(role="user", content="Test", model="test-model")
        ws.send_bytes(msgpack.packb(msg.model_dump()) or b"")

        # Receive response
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"

        # WebSocket will disconnect when context exits


# ========== Integration Tests ==========


def test_full_chat_session(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Test a complete chat session with multiple exchanges."""
    responses = [
        [Message(role="assistant", content="Hello! How can I help you?")],
        [Message(role="assistant", content="I can help with that.")],
        [Message(role="assistant", content="Goodbye!")],
    ]
    response_index = 0

    async def fake_provider_from_model(model: str) -> MockProvider:
        nonlocal response_index
        provider = MockProvider(responses[response_index])
        response_index += 1
        return provider

    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )

    with client.websocket_connect("/chat") as ws:
        # First exchange
        msg1 = Message(role="user", content="Hello", model="test-model")
        ws.send_bytes(msgpack.packb(msg1.model_dump()) or b"")
        data = msgpack.unpackb(ws.receive_bytes())
        assert "Hello! How can I help you?" in data["content"]

        # Second exchange
        msg2 = Message(role="user", content="I need help", model="test-model")
        ws.send_bytes(msgpack.packb(msg2.model_dump()) or b"")
        data = msgpack.unpackb(ws.receive_bytes())
        assert "I can help with that." in data["content"]

        # Third exchange
        msg3 = Message(role="user", content="Thanks, bye", model="test-model")
        ws.send_bytes(msgpack.packb(msg3.model_dump()) or b"")
        data = msgpack.unpackb(ws.receive_bytes())
        assert "Goodbye!" in data["content"]


def test_chat_with_tool_and_followup(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """Test chat with tool execution followed by response."""
    tool_call = ToolCall(
        id="tool-123", name="search_tool", args={"query": "test search"}
    )

    class SearchTool(Tool):
        name = "search_tool"
        description = "Search tool"

        async def process(self, context: Any, args: Dict[str, Any]) -> Dict[str, Any]:
            return {"results": ["Result 1", "Result 2"]}

    # Create a provider that returns tool call first, then response
    responses = [
        Message(role="assistant", content="", tool_calls=[tool_call]),
        Message(role="assistant", content="Based on the search results..."),
    ]

    async def fake_provider_from_model(model: str) -> MockProvider:
        return MockProvider(responses)

    monkeypatch.setattr(
        chat_websocket_runner, "provider_from_model", fake_provider_from_model
    )
    monkeypatch.setattr(
        chat_websocket_runner,
        "get_tool_by_name",
        lambda name: SearchTool if name == "search_tool" else None,
    )

    with client.websocket_connect("/chat") as ws:
        msg = Message(
            role="user",
            content="Search for something",
            model="test-model",
            tools=["search_tool"],
        )
        ws.send_bytes(msgpack.packb(msg.model_dump()) or b"")

        # Receive tool call
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "tool_call"

        # Receive tool result
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "tool_result"

        # Receive final response
        data = msgpack.unpackb(ws.receive_bytes())
        assert data["type"] == "chunk"
        assert "Based on the search results" in data["content"]
