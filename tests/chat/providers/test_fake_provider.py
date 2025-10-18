import pytest

from nodetool.providers.fake_provider import (
    FakeProvider,
    create_fake_tool_call,
    create_simple_fake_provider,
    create_streaming_fake_provider,
    create_tool_calling_fake_provider,
)
from nodetool.metadata.types import Message, MessageTextContent, ToolCall
from nodetool.workflows.types import Chunk


class TestFakeProvider:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """Test basic text response functionality."""
        provider = FakeProvider(text_response="Hello, world!", should_stream=False)

        messages = [Message(role="user", content=[MessageTextContent(text="Hi")])]

        # Test generate_message
        response = await provider.generate_message(messages, "test-model")
        assert response.role == "assistant"
        assert response.content is not None
        assert len(response.content) == 1
        assert isinstance(response.content[0], MessageTextContent)
        assert response.content[0].text == "Hello, world!"

        # Check call tracking
        assert provider.call_count == 1
        assert provider.last_model == "test-model"
        assert provider.last_messages == messages

    @pytest.mark.asyncio
    async def test_streaming_text_response(self):
        """Test streaming text response functionality."""
        provider = FakeProvider(
            text_response="Hello there!", should_stream=True, chunk_size=5
        )

        messages = [Message(role="user", content=[MessageTextContent(text="Hi")])]

        chunks = []
        async for chunk in provider.generate_messages(messages, "test-model"):
            chunks.append(chunk)

        # Should get 3 chunks: "Hello", " ther", "e!"
        assert len(chunks) == 3
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " ther"
        assert chunks[2].content == "e!"
        assert chunks[2].done  # Last chunk should be marked as done
        assert not chunks[0].done  # Earlier chunks should not be done

    @pytest.mark.asyncio
    async def test_non_streaming_text_response(self):
        """Test non-streaming text response functionality."""
        provider = FakeProvider(text_response="Complete response", should_stream=False)

        messages = [Message(role="user", content=[MessageTextContent(text="Hi")])]

        chunks = []
        async for chunk in provider.generate_messages(messages, "test-model"):
            chunks.append(chunk)

        # Should get single chunk
        assert len(chunks) == 1
        assert chunks[0].content == "Complete response"
        assert chunks[0].done

    @pytest.mark.asyncio
    async def test_tool_calls_response(self):
        """Test tool calls response functionality."""
        tool_calls = [
            create_fake_tool_call("search", {"query": "test"}),
            create_fake_tool_call("calculate", {"expression": "2+2"}),
        ]
        provider = FakeProvider(tool_calls=tool_calls)

        messages = [Message(role="user", content=[MessageTextContent(text="Help me")])]

        # Test generate_message with tool calls
        response = await provider.generate_message(messages, "test-model")
        assert response.role == "assistant"
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[1].name == "calculate"

        # Test generate_messages with tool calls
        results = []
        async for result in provider.generate_messages(messages, "test-model"):
            results.append(result)

        assert len(results) == 2
        assert all(isinstance(result, ToolCall) for result in results)
        assert results[0].name == "search"
        assert results[1].name == "calculate"

    @pytest.mark.asyncio
    async def test_custom_response_function(self):
        """Test custom response function functionality."""

        def custom_fn(messages, model):
            # Return different responses based on input
            if "calculate" in str(messages):
                return "The answer is 42"
            else:
                return "I don't understand"

        provider = FakeProvider(custom_response_fn=custom_fn, should_stream=False)

        # Test with calculate query
        calc_messages = [
            Message(role="user", content=[MessageTextContent(text="calculate 2+2")])
        ]
        response = await provider.generate_message(calc_messages, "test-model")
        assert response.content is not None
        assert len(response.content) == 1
        assert isinstance(response.content[0], MessageTextContent)
        assert response.content[0].text == "The answer is 42"

        # Test with other query
        other_messages = [
            Message(role="user", content=[MessageTextContent(text="hello")])
        ]
        response = await provider.generate_message(other_messages, "test-model")
        assert response.content is not None
        assert len(response.content) == 1
        assert isinstance(response.content[0], MessageTextContent)
        assert response.content[0].text == "I don't understand"

    @pytest.mark.asyncio
    async def test_call_count_tracking(self):
        """Test call count tracking functionality."""
        provider = FakeProvider()
        messages = [Message(role="user", content=[MessageTextContent(text="Hi")])]

        assert provider.call_count == 0

        await provider.generate_message(messages, "test-model")
        assert provider.call_count == 1

        async for _ in provider.generate_messages(messages, "test-model"):
            pass
        assert provider.call_count == 2

        provider.reset_call_count()
        assert provider.call_count == 0

    def test_create_fake_tool_call(self):
        """Test tool call creation helper."""
        tool_call = create_fake_tool_call("test_tool", {"arg1": "value1"})

        assert isinstance(tool_call, ToolCall)
        assert tool_call.name == "test_tool"
        assert tool_call.args == {"arg1": "value1"}
        assert tool_call.id is not None  # Should have a UUID

        # Test with custom ID
        custom_tool_call = create_fake_tool_call("custom", call_id="test-id-123")
        assert custom_tool_call.id == "test-id-123"

    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test convenience creation functions."""
        # Test simple provider
        simple = create_simple_fake_provider("Simple response")
        response = await simple.generate_message([], "test")
        assert response.content is not None
        assert len(response.content) == 1
        assert isinstance(response.content[0], MessageTextContent)
        assert response.content[0].text == "Simple response"

        # Test streaming provider
        streaming = create_streaming_fake_provider("12345678", chunk_size=3)
        chunks = []
        async for chunk in streaming.generate_messages([], "test"):
            chunks.append(chunk)
        assert len(chunks) == 3  # "123", "456", "78"

        # Test tool calling provider
        tool_calls = [create_fake_tool_call("test")]
        tool_provider = create_tool_calling_fake_provider(tool_calls)
        results = []
        async for result in tool_provider.generate_messages([], "test"):
            results.append(result)
        assert len(results) == 1
        assert isinstance(results[0], ToolCall)
