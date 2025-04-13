"""
Base provider class for chat completion services.

This module provides the foundation for all chat provider implementations, defining
a common interface that all providers must implement for streaming completions and tool handling.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Sequence

from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import Message, ToolCall, MessageFile
from nodetool.workflows.types import Chunk

import json
import datetime


class ChatProvider(ABC):
    """
    Abstract base class for chat completion providers (OpenAI, Anthropic, Ollama, etc.).

    Defines a common interface for different chat providers, allowing the chat module
    to work with any supported provider interchangeably. Subclasses must implement
    the generate_messages method to define provider-specific behavior.
    """

    log_file: str | None = None

    def __init__(self):
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        }

    def _log_api_request(
        self,
        method: str,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Tool],
        **kwargs,
    ) -> None:
        """Log an API request to the specified log file.

        Args:
            method: The API method being called
            messages: The conversation history
            model: The model to use
            tools: Optional tools to make available to the model
            **kwargs: Additional parameters to pass to the API
        """
        if not self.log_file:
            return

        try:
            with open(self.log_file, "a") as f:
                timestamp = datetime.datetime.now().isoformat()
                log_entry = {
                    "timestamp": timestamp,
                    "type": "request",
                    "method": method,
                    "model": model,
                    "messages": [msg.model_dump() for msg in messages],
                    "tools": [tool.name for tool in tools],
                    **kwargs,
                }
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging API request: {e}")

    def _log_tool_call(self, tool_call: ToolCall) -> None:
        """Log a tool call to the specified log file.

        Args:
            tool_call: The tool call to log
        """
        if not self.log_file:
            return

        try:
            with open(self.log_file, "a") as f:
                timestamp = datetime.datetime.now().isoformat()
                log_entry = {
                    "timestamp": timestamp,
                    "type": "tool_call",
                    "tool_name": tool_call.name,
                    "arguments": tool_call.args,
                }
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging tool call: {e}")

    def _log_api_response(self, method: str, response: Message) -> None:
        """Log an API response to the specified log file.

        Args:
            method: The API method that was called
            response: The response from the API
        """
        if not self.log_file:
            return

        try:
            # Convert response to serializable dict
            response_dict = {
                "role": response.role,
                "content": response.content,
            }

            # Add tool calls if present
            if response.tool_calls:
                response_dict["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.args,
                        }
                    }
                    for tc in response.tool_calls
                ]

            # Log each tool call as a separate entry
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    self._log_tool_call(tool_call)

            with open(self.log_file, "a") as f:
                timestamp = datetime.datetime.now().isoformat()
                log_entry = {
                    "timestamp": timestamp,
                    "type": "response",
                    "method": method,
                    "response": response_dict,
                }
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging API response: {e}")

    @abstractmethod
    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        **kwargs,
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
        **kwargs,
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


class MockProvider(ChatProvider):
    """
    A mock chat provider for testing purposes.

    Allows defining a sequence of responses (text or tool calls) that the
    provider will return upon subsequent calls to generate_message or generate_messages.
    """

    def __init__(self, responses: Sequence[Message], log_file: str | None = None):
        """
        Initialize the MockProvider.

        Args:
            responses: A sequence of Message objects to be returned by generate calls.
                       Each call consumes one message from the sequence.
            log_file: Optional path to a log file.
        """
        super().__init__()
        self.responses = list(responses)  # Store responses
        self.call_log = []  # Log calls made to the provider
        self.response_index = 0
        self.log_file = log_file

    def _get_next_response(self) -> Message:
        """Returns the next predefined response or raises an error if exhausted."""
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        else:
            raise IndexError("MockProvider has run out of predefined responses.")

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        **kwargs,
    ) -> Message:
        """
        Simulates generating a single message.

        Logs the call and returns the next predefined response.
        """
        self._log_api_request("generate_message", messages, model, tools, **kwargs)
        self.call_log.append(
            {
                "method": "generate_message",
                "messages": messages,
                "model": model,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        response = self._get_next_response()
        self._log_api_response("generate_message", response)

        # Simulate output file generation if present in the mock response
        if hasattr(response, "output_files") and response.output_files:
            # This provider doesn't interact with the filesystem,
            # it just returns the message as defined.
            # The SubTaskContext will handle the file saving based on this message.
            pass

        return response

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        **kwargs,
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
        """
        Simulates generating messages, yielding chunks or tool calls.

        Currently yields the entire next predefined response. Can be adapted
        to yield individual chunks/tool calls if needed for more granular testing.
        """
        self._log_api_request("generate_messages", messages, model, tools, **kwargs)
        self.call_log.append(
            {
                "method": "generate_messages",
                "messages": messages,
                "model": model,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        response = self._get_next_response()
        self._log_api_response(
            "generate_messages", response
        )  # Log the full conceptual response

        # Simulate streaming behavior
        if response.tool_calls:
            for tool_call in response.tool_calls:
                self._log_tool_call(tool_call)  # Log individual tool calls
                yield tool_call
        elif response.content:
            # Yield content as a single chunk for simplicity in this mock
            yield Chunk(content=str(response.content))

        # Simulate output file generation if present in the mock response
        if hasattr(response, "output_files") and response.output_files:
            # Similar to generate_message, just pass the info along.
            # SubTaskContext testing should verify file creation.
            pass
