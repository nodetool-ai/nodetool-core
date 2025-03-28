"""
Regular Chat Flow Module

This module handles the regular (non-agent) chat flow for the NodeTool chat CLI.
It processes user input, generates responses, and handles tool calls and their results.
"""

import json
from typing import List, Optional

from nodetool.chat.providers import Chunk
from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.tools import Tool
from nodetool.chat.chat import run_tool, default_serializer
from nodetool.metadata.types import Message, FunctionModel, ToolCall
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.tools import (
    SearchEmailTool,
    GoogleSearchTool,
    AddLabelTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
    BrowserTool,
    ScreenshotTool,
    SearchFileTool,
    ChromaTextSearchTool,
    ChromaHybridSearchTool,
    ExtractPDFTablesTool,
    ExtractPDFTextTool,
    ConvertPDFToMarkdownTool,
    CreateAppleNoteTool,
    ReadAppleNotesTool,
)


async def process_regular_chat(
    user_input: str,
    messages: List[Message],
    model: str,
    provider: ChatProvider,
    workspace_dir: str,
    context: ProcessingContext,
    debug_mode: bool = False,
) -> List[Message]:
    """
    Process a user message in regular chat mode (non-agent).

    Args:
        user_input: The text input from the user
        messages: The current message history
        model: The AI model to use
        tools: Available tools
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
        SearchEmailTool(workspace_dir),
        GoogleSearchTool(workspace_dir),
        AddLabelTool(workspace_dir),
        ListDirectoryTool(workspace_dir),
        ReadFileTool(workspace_dir),
        WriteFileTool(workspace_dir),
        BrowserTool(workspace_dir),
        ScreenshotTool(workspace_dir),
        SearchFileTool(workspace_dir),
        ExtractPDFTablesTool(workspace_dir),
        ExtractPDFTextTool(workspace_dir),
        ConvertPDFToMarkdownTool(workspace_dir),
        CreateAppleNoteTool(workspace_dir),
        ReadAppleNotesTool(workspace_dir),
    ]

    while True:
        async for chunk in provider.generate_messages(
            messages=messages_to_send,
            model=model,
            tools=tools,
            provider=provider,
        ):  # type: ignore
            if isinstance(chunk, Chunk):
                current_chunk = str(chunk.content)
                print(chunk.content, end="", flush=True)
                if messages[-1].role == "assistant":
                    assert isinstance(messages[-1].content, str)
                    messages[-1].content += current_chunk
                else:
                    messages.append(Message(role="assistant", content=current_chunk))
                if chunk.done:
                    print("")

            if isinstance(chunk, ToolCall):
                # Display tool call in debug mode
                if debug_mode:
                    print("\n[Debug] Tool Call:")
                    print(f"  Name: {chunk.name}")
                    print(f"  Arguments: {json.dumps(chunk.args, indent=2)}")

                tool_result = await run_tool(context, chunk, tools)

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
