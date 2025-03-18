"""
Base provider class for chat completion services.

This module provides the foundation for all chat provider implementations, defining
a common interface that all providers must implement for streaming completions and tool handling.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Sequence

from nodetool.metadata.types import FunctionModel, Message, ToolCall
from nodetool.workflows.processing_context import ProcessingContext


class Chunk:
    """A chunk of streamed content from a provider."""

    def __init__(self, content: str, done: bool = False):
        self.content = content
        self.done = done


class ChatProvider(ABC):
    """
    Abstract base class for chat completion providers (OpenAI, Anthropic, Ollama, etc.).

    Defines a common interface for different chat providers, allowing the chat module
    to work with any supported provider interchangeably. Subclasses must implement
    the generate_messages method to define provider-specific behavior.
    """

    def __init__(self):
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }

    @abstractmethod
    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        **kwargs
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
        """
        Generate message completions from the provider, yielding chunks or tool calls.

        Args:
            messages: Sequence of Message objects representing the conversation
            model: str containing model information
            tools: Available tools for the model to use
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunk objects with content and completion status or ToolCall objects
        """
        pass

    @abstractmethod
    def convert_message(self, message: Message) -> Any:
        """
        Convert an internal message to the provider-specific format.

        Args:
            message: The Message object to convert

        Returns:
            The message in the provider's expected format
        """
        pass

    @abstractmethod
    def format_tools(self, tools: Sequence[Any]) -> list:
        """
        Convert tool definitions to the provider-specific format.

        Args:
            tools: List of Tool instances

        Returns:
            List of tools in the provider's expected format
        """
        pass
