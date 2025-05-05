"""
Regular Chat Flow Module

This module handles the regular (non-agent) chat flow for the NodeTool chat CLI.
It processes user input, generates responses, and handles tool calls and their results.
"""

import json
from typing import List, Optional, Sequence

from nodetool.agents.tools.base import Tool
from nodetool.ui import console
from nodetool.workflows.types import Chunk
from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.chat import default_serializer
from nodetool.metadata.types import Message, ToolCall
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools import (
    AddLabelTool,
    ArchiveEmailTool,
    BrowserTool,
    ConvertPDFToMarkdownTool,
    DownloadFileTool,
    ExtractPDFTablesTool,
    ExtractPDFTextTool,
    GoogleGroundedSearchTool,
    GoogleImageGenerationTool,
    GoogleImagesTool,
    GoogleNewsTool,
    GoogleSearchTool,
    ListAssetsDirectoryTool,
    OpenAIImageGenerationTool,
    OpenAITextToSpeechTool,
    OpenAIWebSearchTool,
    ReadAssetTool,
    SaveAssetTool,
    ScreenshotTool,
    SearchEmailTool,
)
from rich.status import Status


async def run_tool(
    status: Status,
    context: ProcessingContext,
    tool_call: ToolCall,
    tools: Sequence[Tool],
) -> ToolCall:
    """Execute a tool call requested by the chat model.

    Locates the appropriate tool implementation by name from the available tools,
    executes it with the provided arguments, and captures the result.

    Args:
        console (Console): The console to use for status updates
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

    status.update(tool.user_message(tool_call.args), spinner="dots")
    result = await tool.process(context, tool_call.args)

    return ToolCall(
        id=tool_call.id,
        name=tool_call.name,
        args=tool_call.args,
        result=result,
    )


async def process_regular_chat(
    user_input: str,
    messages: List[Message],
    model: str,
    provider: ChatProvider,
    status: Status,
    context: ProcessingContext,
    debug_mode: bool = False,
) -> List[Message]:
    """
    Process a user message in regular chat mode (non-agent).

    Args:
        user_input: The text input from the user
        messages: The current message history
        model: The AI model to use
        provider: The chat provider to use
        console: The console to use for status updates
        context: The processing context
        debug_mode: Whether to display debug information about tool calls

    Returns:
        Updated message history
    """
    # Add user message
    messages.append(Message(role="user", content=user_input))
    unprocessed_messages = []

    # Process messages
    messages_to_send = messages

    tools: List[Tool] = [
        AddLabelTool(),
        ArchiveEmailTool(),
        BrowserTool(),
        ConvertPDFToMarkdownTool(),
        DownloadFileTool(),
        ExtractPDFTablesTool(),
        ExtractPDFTextTool(),
        GoogleGroundedSearchTool(),
        GoogleImageGenerationTool(),
        GoogleImagesTool(),
        GoogleNewsTool(),
        GoogleSearchTool(),
        ListAssetsDirectoryTool(),
        OpenAIImageGenerationTool(),
        OpenAITextToSpeechTool(),
        OpenAIWebSearchTool(),
        ReadAssetTool(),
        SaveAssetTool(),
        ScreenshotTool(),
        SearchEmailTool(),
    ]

    while True:
        async for chunk in provider.generate_messages(
            messages=messages_to_send,
            model=model,
            tools=tools,
        ):  # type: ignore
            if isinstance(chunk, Chunk):
                current_chunk = str(chunk.content)
                print(chunk.content, end="")
                if messages[-1].role == "assistant":
                    assert isinstance(messages[-1].content, str)
                    messages[-1].content += current_chunk
                else:
                    messages.append(Message(role="assistant", content=current_chunk))

            if isinstance(chunk, ToolCall):
                # Display tool call in debug mode
                if debug_mode:
                    print("\n[Debug] Tool Call:")
                    print(f"  Name: {chunk.name}")
                    print(f"  Arguments: {json.dumps(chunk.args, indent=2)}")

                tool_result = await run_tool(
                    status=status,
                    context=context,
                    tool_call=chunk,
                    tools=tools,
                )

                # Display tool result in debug mode
                if debug_mode:
                    print("[Debug] Tool Result:")
                    print(
                        f"  {json.dumps(tool_result.result, indent=2, default=default_serializer)}\n"
                    )

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

        # If there are unprocessed messages, continue the conversation
        if unprocessed_messages:
            messages.extend(unprocessed_messages)
            messages_to_send = unprocessed_messages
            unprocessed_messages = []
        else:
            break

    return messages
