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
from typing import Any, Sequence

import openai
from pydantic import BaseModel
from rich.console import Console

from nodetool.agents.tools.base import Tool
from nodetool.common.environment import Environment
from nodetool.metadata.types import (
    Message,
    OpenAIModel,
    ToolCall,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


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
