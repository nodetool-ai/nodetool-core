"""
Base provider class for chat completion services.

This module provides the foundation for all chat provider implementations, defining
a common interface that all providers must implement for streaming completions and tool handling.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Sequence

from nodetool.metadata.types import Message, ToolCall
from nodetool.workflows.types import Chunk


class ChatProvider(ABC):
    """
    Abstract base class for chat completion providers (OpenAI, Anthropic, Ollama, etc.).

    Defines a common interface for different chat providers, allowing the chat module
    to work with any supported provider interchangeably. Subclasses must implement
    the generate_messages method to define provider-specific behavior.
    """

    has_code_interpreter: bool = False

    def __init__(self):
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }

    @abstractmethod
    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        use_code_interpreter: bool = False,
        **kwargs
    ) -> Message:
        """
        Generate a single message completion from the provider.

        Args:
            messages: Sequence of Message objects representing the conversation
            model: str containing model information
            tools: Available tools for the model to use
            **kwargs: Additional provider-specific parameters

        Returns:
            A message returned by the provider.
        """
        pass

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
