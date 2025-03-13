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
import json
import traceback
from typing import Any, AsyncGenerator, Sequence

import openai
from pydantic import BaseModel

from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools import (
    AddLabelTool,
    BrowserTool,
    ChromaHybridSearchTool,
    ChromaTextSearchTool,
    CreateAppleNoteTool,
    ExtractPDFTablesTool,
    ExtractPDFTextTool,
    ConvertPDFToMarkdownTool,
    FindNodeTool,
    KeywordDocSearchTool,
    ReadAppleNotesTool,
    ScreenshotTool,
    SearchEmailTool,
    SearchFileTool,
    SemanticDocSearchTool,
    Tool,
)

from nodetool.common.environment import Environment
from nodetool.metadata.types import (
    ColumnDef,
    FunctionModel,
    Message,
    OpenAIModel,
    Provider,
    ToolCall,
)
from nodetool.chat.ollama_service import get_ollama_models
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.tools import ListDirectoryTool, ReadFileTool, WriteFileTool
import readline
import os


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


AVAILABLE_CHAT_TOOLS = [
    SearchEmailTool(),
    AddLabelTool(),
    ListDirectoryTool(),
    ReadFileTool(),
    WriteFileTool(),
    BrowserTool(),
    ScreenshotTool(),
    SearchFileTool(),
    ChromaTextSearchTool(),
    ChromaHybridSearchTool(),
    ExtractPDFTablesTool(),
    ExtractPDFTextTool(),
    ConvertPDFToMarkdownTool(),
    CreateAppleNoteTool(),
    ReadAppleNotesTool(),
    SemanticDocSearchTool(),
    KeywordDocSearchTool(),
]

AVAILABLE_CHAT_TOOLS_BY_NAME = {tool.name: tool for tool in AVAILABLE_CHAT_TOOLS}


def json_schema_for_column(column: ColumnDef) -> dict:
    """Create a JSON schema for a database column definition.

    Converts a ColumnDef object to a JSON schema representation that can be used in JSON schema
    validation. Different data types are mapped to appropriate JSON schema types with format
    specifications where applicable.

    Args:
        column (ColumnDef): The column definition containing name, data type, and description

    Returns:
        dict: A JSON schema object representing the column with type and description

    Raises:
        ValueError: If an unsupported data type is encountered
    """
    data_type = column.data_type
    description = column.description or ""

    if data_type == "string":
        return {"type": "string", "description": description}
    if data_type == "number":
        return {"type": "number", "description": description}
    if data_type == "int":
        return {"type": "integer", "description": description}
    if data_type == "float":
        return {"type": "number", "description": description}
    if data_type == "datetime":
        return {"type": "string", "format": "date-time", "description": description}
    raise ValueError(f"Unknown data type {data_type}")


def json_schema_for_dataframe(columns: list[ColumnDef]) -> dict:
    """Create a JSON schema for a DataFrame.

    Builds a comprehensive JSON schema that represents a DataFrame structure with
    the specified columns. The schema enforces that all required columns are present
    and prevents additional properties.

    Args:
        columns (list[ColumnDef]): List of column definitions for the DataFrame

    Returns:
        dict: A JSON schema object with nested properties representing the DataFrame structure
            with a "data" array containing objects that conform to the column definitions
    """
    return {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        column.name: json_schema_for_column(column)
                        for column in columns
                    },
                    "required": [column.name for column in columns],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["data"],
        "additionalProperties": False,
    }


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
    model: FunctionModel,
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
    ):
        yield chunk


async def process_messages(
    messages: Sequence[Message],
    model: FunctionModel,
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


async def chat_cli():
    """Interactive command-line chat interface with multi-provider support.

    Provides a CLI interface for interacting with various AI models through different providers.
    Features include:

    - Support for OpenAI, Anthropic, and Ollama providers
    - Model selection and provider switching
    - Command history with readline integration
    - Tab completion for commands and models
    - Tool execution capabilities
    - Streaming responses with real-time display

    Commands:
        /provider [openai|anthropic|ollama]: Switch between AI providers
        /model [model_name]: Select a specific model
        /models: List available models for the current provider
        /clear: Clear the conversation history
        /quit or /exit: Exit the chat interface
        /help: Display available commands

    The CLI maintains conversation context and supports tool execution requested by AI models.
    Command history is preserved between sessions in ~/.nodetool_history.
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    context = ProcessingContext(user_id="test", auth_token="test")
    ollama_models = await get_ollama_models()
    openai_models = await get_openai_models()

    # Define default models for each provider
    default_models = {
        Provider.OpenAI: "gpt-4o",
        Provider.Anthropic: "claude-3-5-sonnet-20241022",
        Provider.Ollama: "llama3.2:3b",
    }

    provider = Provider.OpenAI
    model = FunctionModel(name=default_models[provider], provider=provider)
    messages: list[Message] = []

    # Set up readline
    histfile = os.path.join(os.path.expanduser("~"), ".nodetool_history")
    try:
        readline.read_history_file(histfile)
    except FileNotFoundError:
        pass

    # Enable tab completion
    COMMANDS = ["help", "quit", "exit", "provider", "model", "models", "clear"]
    PROVIDERS = [p.value.lower() for p in Provider]

    def completer(text, state):
        if text.startswith("/"):
            text = text[1:]
            options = [f"/{cmd}" for cmd in COMMANDS if cmd.startswith(text)]
        elif text.startswith("/model "):
            model_text = text.split()[-1]
            options = [
                f"/model {m}" for m in ollama_models if m.name.startswith(model_text)
            ]
            options.extend(
                [f"/model {m}" for m in openai_models if m.id.startswith(model_text)]
            )
        elif text.startswith("/provider "):
            provider_text = text.split()[-1]
            options = [
                f"/provider {p}" for p in PROVIDERS if p.startswith(provider_text)
            ]
        else:
            return None
        return options[state] if state < len(options) else None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")

    print("Chat CLI - Type /help for commands")

    while True:
        try:
            user_input = input("> ").strip()
            readline.write_history_file(histfile)

            if user_input.startswith("/"):
                cmd = user_input[1:].lower()
                if cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "help":
                    print("Commands:")
                    print("  /provider [openai|anthropic|ollama] - Set the provider")
                    print("  /model [model_name] - Set the model")
                    print("  /models - List available Ollama models")
                    print("  /clear - Clear chat history")
                    print("  /quit or /exit - Exit the chat")
                    continue
                elif cmd == "models":
                    try:
                        print("\nAvailable Ollama models:")
                        for model_info in ollama_models:
                            print(f"- {model_info.name}")
                        print()
                    except Exception as e:
                        print(f"Error listing models: {e}")
                    continue
                elif cmd.startswith("provider "):
                    provider_name = cmd.split()[1].capitalize()
                    try:
                        provider = Provider[provider_name]
                        # Update model to default for new provider
                        model = FunctionModel(
                            name=default_models[provider], provider=provider
                        )
                        print(
                            f"Provider set to {provider.value} with default model {default_models[provider]}"
                        )
                    except KeyError:
                        print(
                            f"Invalid provider. Choose from: {[p.value for p in Provider]}"
                        )
                    continue
                elif cmd.startswith("model "):
                    model_name = cmd.split()[1]
                    model = FunctionModel(name=model_name, provider=provider)
                    print(f"Model set to {model_name}")
                    continue
                elif cmd == "clear":
                    messages = []
                    print("Chat history cleared")
                    continue
                else:
                    print("Unknown command. Type /help for available commands")
                    continue

            # Add user message
            messages.append(Message(role="user", content=user_input))
            unprocessed_messages = messages

            while unprocessed_messages:
                messages_to_send = messages + unprocessed_messages
                unprocessed_messages = []
                async for chunk in generate_messages(
                    messages=messages_to_send,
                    model=model,
                    tools=AVAILABLE_CHAT_TOOLS,
                ):
                    if isinstance(chunk, Chunk):
                        current_chunk = str(chunk.content)
                        print(chunk.content, end="", flush=True)
                        if messages[-1].role == "assistant":
                            assert isinstance(messages[-1].content, str)
                            messages[-1].content += current_chunk
                        else:
                            messages.append(
                                Message(role="assistant", content=current_chunk)
                            )
                        if chunk.done:
                            print("")

                    if isinstance(chunk, ToolCall):
                        # print(f"Running {chunk.name} with {chunk.args}")
                        tool_result = await run_tool(
                            context, chunk, AVAILABLE_CHAT_TOOLS
                        )
                        # print(tool_result)
                        unprocessed_messages.append(
                            Message(role="assistant", tool_calls=[chunk])
                        )
                        unprocessed_messages.append(
                            Message(
                                role="tool",
                                tool_call_id=tool_result.id,
                                name=chunk.name,
                                content=json.dumps(
                                    tool_result.result, default=default_serializer
                                ),
                            )
                        )

        except KeyboardInterrupt:
            return
        except EOFError:
            return
        except Exception as e:
            print(f"Error: {e}")
            stacktrace = traceback.format_exc()
            print(f"Stacktrace: {stacktrace}")


if __name__ == "__main__":
    asyncio.run(chat_cli())
