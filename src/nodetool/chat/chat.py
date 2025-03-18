"""
Chat module providing multi-provider chat functionality with tool integration.

This module implements a chat interface that supports multiple AI providers (OpenAI, Anthropic, Ollama)
and allows for tool-augmented conversations. It handles:

- Message conversion between different provider formats
- Streaming chat completions
- Tool execution and integration
- CLI interface for interactive chat
- Provider-specific client management

The module supports various content types including text and images, and provides
a unified interface for handling tool calls across different providers.

Key components:
- Provider implementations for each service
- Streaming completion handlers
- Tool execution framework
- Interactive CLI with command history and tab completion
"""

import asyncio
from typing import Any, AsyncGenerator, Sequence

import openai
from pydantic import BaseModel

from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools.base import Tool
from nodetool.common.environment import Environment
from nodetool.metadata.types import (
    Message,
    OpenAIModel,
    ToolCall,
)
from nodetool.workflows.processing_context import ProcessingContext


async def get_openai_models():
    """Get available OpenAI models.

    Retrieves a list of available models from the OpenAI API using the configured API key
    from the environment. The models are returned as OpenAIModel objects with essential
    metadata.

    Returns:
        list[OpenAIModel]: A list of available OpenAI models with their metadata including
            id, object type, creation timestamp, and owner information.

    Raises:
        AssertionError: If OPENAI_API_KEY is not set in the environment.
        openai.OpenAIError: If there's an error connecting to the OpenAI API.
    """
    env = Environment.get_environment()
    api_key = env.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY is not set"

    client = openai.AsyncClient(api_key=api_key)
    res = await client.models.list()
    return [
        OpenAIModel(
            id=model.id,
            object=model.object,
            created=model.created,
            owned_by=model.owned_by,
        )
        for model in res.data
    ]


def default_serializer(obj: Any) -> dict:
    """Serialize Pydantic models to dict.

    Custom serializer for JSON encoding that handles Pydantic models by converting them
    to dictionaries. Used for serializing complex objects during tool operations.

    Args:
        obj (Any): The object to serialize

    Returns:
        dict: Dictionary representation of the Pydantic model

    Raises:
        TypeError: If the object is not a Pydantic model and cannot be serialized
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    raise TypeError("Type not serializable")


async def generate_messages(
    messages: Sequence[Message],
    model: str,
    tools: Sequence[Tool] = [],
    **kwargs,
) -> AsyncGenerator[Chunk | ToolCall, Any]:
    """
    Generate messages using the appropriate provider for the model.

    This function dispatches to the correct provider based on the model's provider field
    and yields streamed chunks and tool calls.

    Args:
        messages: Sequence of Message objects representing the conversation
        model: Function model containing name and provider
        tools: Available tools for the model to use
        **kwargs: Additional provider-specific parameters

    Yields:
        Chunk objects with content or ToolCall objects
    """
    provider = get_provider(model.provider)
    async for chunk in provider.generate_messages(
        messages=messages,
        model=model,
        tools=tools,
        **kwargs,
    ):  # type: ignore
        yield chunk


async def process_messages(
    messages: Sequence[Message],
    model: str,
    tools: Sequence[Tool] = [],
    **kwargs,
) -> Message:
    """
    Process messages and return a single accumulated response message.

    Args:
        messages: The messages to process
        model: The model to use
        tools: Available tools
        **kwargs: Additional arguments passed to the model

    Returns:
        Message: The complete response message with content and tool calls
    """
    content = ""
    tool_calls: list[ToolCall] = []

    async for chunk in generate_messages(messages, model, tools, **kwargs):
        if isinstance(chunk, Chunk):
            content += chunk.content
        elif isinstance(chunk, ToolCall):
            tool_calls.append(chunk)

    return Message(
        role="tool" if tool_calls else "assistant",
        content=content if content else None,
        tool_calls=tool_calls if tool_calls else None,
    )


async def run_tool(
    context: ProcessingContext,
    tool_call: ToolCall,
    tools: Sequence[Tool],
) -> ToolCall:
    """Execute a tool call requested by the chat model.

    Locates the appropriate tool implementation by name from the available tools,
    executes it with the provided arguments, and captures the result.

    Args:
        context (ProcessingContext): The processing context containing user information and state
        tool_call (ToolCall): The tool call to execute, containing name, ID, and arguments
        tools (Sequence[Tool]): Available tools that can be executed

    Returns:
        ToolCall: The original tool call object updated with the execution result

    Raises:
        AssertionError: If the specified tool is not found in the available tools
    """

    def find_tool(name):
        for tool in tools:
            if tool.name == name:
                return tool
        return None

    tool = find_tool(tool_call.name)

    assert tool is not None, f"Tool {tool_call.name} not found"

    result = await tool.process(context, tool_call.args)

    return ToolCall(
        id=tool_call.id,
        name=tool_call.name,
        args=tool_call.args,
        result=result,
    )


async def run_tools(
    context: ProcessingContext,
    tool_calls: Sequence[ToolCall],
    tools: Sequence[Tool],
) -> list[ToolCall]:
    """Execute a list of tool calls in parallel.

    Runs multiple tool calls concurrently using asyncio.gather to improve performance when
    multiple tools need to be executed. Each tool call is processed independently.

    Args:
        context (ProcessingContext): The processing context containing user information and state
        tool_calls (Sequence[ToolCall]): A sequence of tool calls to execute
        tools (Sequence[Tool]): Available tools that can be executed

    Returns:
        list[ToolCall]: List of tool calls with their execution results
    """
    return await asyncio.gather(
        *[
            run_tool(
                context=context,
                tool_call=tool_call,
                tools=tools,
            )
            for tool_call in tool_calls
        ]
    )
