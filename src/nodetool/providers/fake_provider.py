"""
Fake provider implementation for easy testing.

This module provides a simplified testing provider that can return configurable
responses without requiring predefined message sequences. Ideal for unit tests.

Example usage:

    # Simple text response
    provider = create_simple_fake_provider("Test response")

    # Streaming response
    provider = create_streaming_fake_provider("Hello world", chunk_size=5)

    # Tool calling
    tool_calls = [create_fake_tool_call("search", {"query": "test"})]
    provider = create_tool_calling_fake_provider(tool_calls)

    # Custom logic based on input
    def smart_response(messages, model):
        if "math" in str(messages):
            return "42"
        return "I don't know"

    provider = FakeProvider(custom_response_fn=smart_response)

    # Use in tests
    with patch("module.get_provider", return_value=provider):
        # test code here
"""

import uuid
from typing import Any, AsyncGenerator, Callable, List, Sequence, Union

from nodetool.metadata.types import (
    LanguageModel,
    Message,
    MessageTextContent,
    ToolCall,
)
from nodetool.providers.base import BaseProvider
from nodetool.workflows.types import Chunk


class FakeProvider(BaseProvider):
    """
    A simplified fake chat provider for testing.

    Unlike MockProvider which requires predefined responses, FakeProvider allows
    configuring simple text responses, tool calls, or custom response functions
    on the fly. Perfect for unit tests that need predictable behavior.
    """

    provider_name: str = "fake"

    def __init__(
        self,
        text_response: str = "Hello, this is a fake response!",
        tool_calls: list[ToolCall] | None = None,
        should_stream: bool = True,
        chunk_size: int = 10,
        custom_response_fn: (Callable[[Sequence[Message], str], str | list[ToolCall]] | None) = None,
    ):
        """
        Initialize the FakeProvider.

        Args:
            text_response: Default text to return (if no custom_response_fn)
            tool_calls: Optional list of tool calls to return instead of text
            should_stream: Whether to stream response in chunks or return all at once
            chunk_size: Number of characters per chunk when streaming text
            custom_response_fn: Optional function that takes (messages, model) and returns
                               either a string or list[ToolCall]
        """
        super().__init__()
        self.text_response = text_response
        self.tool_calls = tool_calls or []
        self.should_stream = should_stream
        self.chunk_size = chunk_size
        self.custom_response_fn = custom_response_fn
        self.call_count = 0
        self.last_messages: Sequence[Message] | None = None
        self.last_model: str | None = None
        self.last_tools: Sequence[Any] = []
        self.last_kwargs: dict[str, Any] = {}

    def get_response(self, messages: Sequence[Message], model: str) -> str | list[ToolCall]:
        """Get the response to return (text or tool calls)."""
        if self.custom_response_fn:
            return self.custom_response_fn(messages, model)
        elif self.tool_calls:
            return self.tool_calls
        else:
            return self.text_response

    def reset_call_count(self) -> None:
        """Reset the call count to 0."""
        self.call_count = 0

    async def get_available_language_models(self) -> List[LanguageModel]:
        """Fake provider has no models."""
        return []

    async def generate_message(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        """
        Generate a single message response.

        Returns a Message containing either text content or tool calls
        based on the provider configuration.
        """
        self.call_count += 1
        self.last_messages = messages
        self.last_model = model
        self.last_tools = tools
        self.last_kwargs = kwargs

        response = self.get_response(messages, model)

        if isinstance(response, list):  # Tool calls
            return Message(
                role="assistant",
                content=[],
                tool_calls=response,
            )
        else:  # Text response
            return Message(role="assistant", content=[MessageTextContent(text=response)])

    async def generate_messages(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
        """
        Generate streaming message responses.

        Yields either Chunk objects (for text) or ToolCall objects,
        optionally breaking text into smaller chunks for streaming simulation.
        """
        self.call_count += 1
        self.last_messages = messages
        self.last_model = model
        self.last_tools = tools
        self.last_kwargs = kwargs

        response = self.get_response(messages, model)

        if isinstance(response, list):  # Tool calls
            for tool_call in response:
                yield tool_call
        else:  # Text response
            if self.should_stream and len(response) > self.chunk_size:
                # Break into chunks
                for i in range(0, len(response), self.chunk_size):
                    chunk_text = response[i : i + self.chunk_size]
                    is_done = i + self.chunk_size >= len(response)
                    yield Chunk(content=chunk_text, done=is_done, content_type="text")
            else:
                # Return as single chunk
                yield Chunk(content=response, done=True, content_type="text")


def create_fake_tool_call(
    name: str,
    args: dict[str, Any] | None = None,
    call_id: str | None = None,
) -> ToolCall:
    """
    Convenience function to create a ToolCall for testing.

    Args:
        name: Name of the tool
        args: Arguments dictionary (defaults to empty dict)
        call_id: Tool call ID (generates random UUID if not provided)

    Returns:
        ToolCall object ready for use with FakeProvider
    """
    return ToolCall(
        id=call_id or str(uuid.uuid4()),
        name=name,
        args=args or {},
    )


def create_simple_fake_provider(response_text: str = "Test response") -> FakeProvider:
    """
    Create a FakeProvider with a simple text response.

    Args:
        response_text: The text to return

    Returns:
        Configured FakeProvider
    """
    return FakeProvider(text_response=response_text, should_stream=False)


def create_streaming_fake_provider(
    response_text: str = "This is a streaming test response",
    chunk_size: int = 5,
) -> FakeProvider:
    """
    Create a FakeProvider that streams responses in chunks.

    Args:
        response_text: The text to stream
        chunk_size: Number of characters per chunk

    Returns:
        Configured FakeProvider that streams responses
    """
    return FakeProvider(
        text_response=response_text,
        should_stream=True,
        chunk_size=chunk_size,
    )


def create_tool_calling_fake_provider(tool_calls: list[ToolCall]) -> FakeProvider:
    """
    Create a FakeProvider that returns tool calls.

    Args:
        tool_calls: List of ToolCall objects to return

    Returns:
        Configured FakeProvider that returns tool calls
    """
    return FakeProvider(tool_calls=tool_calls)
