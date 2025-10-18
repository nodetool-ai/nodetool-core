"""
Regular chat processor module.

This module provides the processor for standard chat messages without workflows
or special modes.
"""

from nodetool.config.logging_config import get_logger
import json
import asyncio
from typing import List, Optional
import httpx
from nodetool.agents.tools.tool_registry import resolve_tool_by_name
from pydantic import BaseModel

from nodetool.metadata.types import (
    Message,
    MessageTextContent,
    ToolCall,
)
from nodetool.workflows.types import (
    Chunk,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.graph import Graph
from nodetool.providers.base import BaseProvider
from .base import MessageProcessor

log = get_logger(__name__)
# log.setLevel(logging.DEBUG)

REGULAR_SYSTEM_PROMPT = """
You are a helpful assistant.

Operating mode:
- If something is ambiguous, choose the most reasonable assumption, proceed.
- Prefer tool calls and concrete actions over clarifying questions.

Control of eagerness:
- Keep scope tightly focused.
- Avoid unnecessary exploration.
- Be concise and minimize tokens.

Tool preambles:
- Before each tool call, emit a one-sentence assistant message describing what you're doing and why.

# IMAGE TOOLS
When using image tools, you will get an image url as result.
ALWAYS EMBED THE IMAGE AS MARKDOWN IMAGE TAG.

# File types
References to documents, images, videos or audio files are objects with following structure:
- type: either document, image, video, audio
- uri: either local "file:///path/to/file" or "http://"

# Date and time
Date and time are objects with following structure:
- type: either date, datetime
- year: int
- month: int
- day: int
- hour: int (optional)
- minute: int (optional)
- second: int (optional)
"""


class RegularChatProcessor(MessageProcessor):
    """
    Processor for standard chat messages without workflows or special modes.

    This processor handles regular conversational interactions, including:
    - Text generation with streaming
    - Tool execution
    - Collection context queries
    """

    def __init__(self, provider: BaseProvider):
        super().__init__()
        self.provider = provider

    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        collections: Optional[List[str]] = None,
        graph: Optional[Graph] = None,
        **kwargs,
    ):
        """Process regular chat messages with optional collection context."""
        last_message = chat_history[-1]
        content = ""
        unprocessed_messages = []

        print("last_message", last_message)

        if last_message.tools:
            tools = await asyncio.gather(
                *[
                    resolve_tool_by_name(name, processing_context.user_id)
                    for name in last_message.tools
                ]
            )
        else:
            tools = []

        # Extract query text for collection search
        query_text = self._extract_query_text(last_message)

        if chat_history[0].role != "system":
            chat_history = [
                Message(role="system", content=REGULAR_SYSTEM_PROMPT)
            ] + chat_history

        # Query collections if specified
        collection_context = ""
        if collections:
            log.debug(f"Querying collections: {collections}")
            collection_context = await self._query_collections(
                collections, query_text, n_results=5
            )
            if collection_context:
                log.debug(
                    f"Retrieved collection context: {len(collection_context)} characters"
                )

        assert last_message.model, "Model is required"

        try:
            # Stream the response chunks
            while True:
                messages_to_send = chat_history + unprocessed_messages

                # Add collection context if available
                if collection_context:
                    messages_to_send = self._add_collection_context(
                        messages_to_send, collection_context
                    )
                    collection_context = ""  # Clear after first use

                unprocessed_messages = []

                log.debug(
                    f"Calling provider.generate_messages with {len(messages_to_send)} messages"
                )
                async for chunk in self.provider.generate_messages(
                    messages=messages_to_send,
                    model=last_message.model,
                    tools=tools,
                    context_window=32000,
                ):  # type: ignore
                    if isinstance(chunk, Chunk):
                        content += chunk.content
                        await self.send_message(chunk.model_dump())
                    elif isinstance(chunk, ToolCall):
                        log.debug(f"Processing tool call: {chunk.name}")

                        # Process the tool call
                        tool_result, message = await self._run_tool(
                            processing_context, chunk, graph
                        )
                        log.debug(
                            f"Tool {chunk.name} execution complete, id={tool_result.id}"
                        )

                        # Add tool messages to unprocessed messages
                        assistant_msg = Message(
                            role="assistant",
                            tool_calls=[
                                ToolCall(
                                    id=chunk.id,
                                    name=chunk.name,
                                    args=chunk.args,
                                    result=None,
                                    message=message,
                                )
                            ],
                            thread_id=last_message.thread_id,
                            workflow_id=last_message.workflow_id,
                            provider=last_message.provider,
                            model=last_message.model,
                            agent_mode=last_message.agent_mode or False,
                        )
                        unprocessed_messages.append(assistant_msg)
                        # Stream assistant tool-call message to client so UI can render it immediately
                        await self.send_message(assistant_msg.model_dump())

                        # Convert result to JSON
                        converted_result = self._recursively_model_dump(
                            tool_result.result
                        )
                        tool_result_json = json.dumps(converted_result)
                        tool_msg = Message(
                            role="tool",
                            tool_call_id=tool_result.id,
                            name=chunk.name,
                            content=tool_result_json,
                            thread_id=last_message.thread_id,
                            workflow_id=last_message.workflow_id,
                            provider=last_message.provider,
                            model=last_message.model,
                        )
                        unprocessed_messages.append(tool_msg)
                        # Stream tool result to client for immediate visualization
                        await self.send_message(tool_msg.model_dump())

                # If no more unprocessed messages, we're done
                if not unprocessed_messages:
                    log.debug("No more unprocessed messages, completing generation")
                    break
                else:
                    log.debug(
                        f"Have {len(unprocessed_messages)} unprocessed messages, continuing loop"
                    )

            # Signal completion
            await self.send_message({"type": "chunk", "content": "", "done": True})
            await self.send_message(
                Message(
                    role="assistant",
                    content=content if content else None,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=last_message.agent_mode or False,
                ).model_dump()
            )

        except httpx.ConnectError as e:
            # Handle connection errors
            error_msg = self._format_connection_error(e)
            log.error(f"httpx.ConnectError in process: {e}", exc_info=True)

            # Send error message to client
            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "connection_error",
                }
            )

            # Signal completion even on error
            await self.send_message({"type": "chunk", "content": "", "done": True})

            # Return an error message
            await self.send_message(
                Message(
                    role="assistant",
                    content=f"I encountered a connection error: {error_msg}. Please check your network connection and try again.",
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=last_message.agent_mode or False,
                ).model_dump()
            )

        finally:
            # Always mark processing as complete
            self.is_processing = False

    def _extract_query_text(self, message: Message) -> str:
        """Extract query text from a message."""
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list) and message.content:
            for content_item in message.content:
                if isinstance(content_item, MessageTextContent):
                    return content_item.text
        return ""

    async def _query_collections(
        self, collections: List[str], query_text: str, n_results: int
    ) -> str:
        """Query ChromaDB collections and return concatenated results."""
        if not collections or not query_text:
            return ""

        from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
            get_async_collection,
        )

        all_results = []

        for collection_name in collections:
            collection = await get_async_collection(name=collection_name)
            results = await collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas"],
            )
            print(results)

            if results["documents"] and results["documents"][0]:
                collection_results = f"\n\n### Results from {collection_name}:\n"
                for doc, metadata in zip(
                    results["documents"][0],
                    (
                        results["metadatas"][0]
                        if results["metadatas"]
                        else [{}] * len(results["documents"][0])
                    ),
                ):
                    doc_preview = f"{doc[:200]}..." if len(doc) > 200 else doc
                    collection_results += f"\n- {doc_preview}"
                all_results.append(collection_results)

        return "\n".join(all_results) if all_results else ""

    def _add_collection_context(
        self, messages: List[Message], collection_context: str
    ) -> List[Message]:
        """Add collection context as a system message before the last user message."""
        # Find the last user message index
        last_user_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                last_user_index = i
                break

        if last_user_index >= 0:
            # Insert collection context before the last user message
            collection_message = Message(
                role="system",
                content=f"Context from knowledge base:\n{collection_context}",
            )
            return (
                messages[:last_user_index]
                + [collection_message]
                + messages[last_user_index:]
            )
        return messages

    async def _run_tool(
        self,
        context: ProcessingContext,
        tool_call: ToolCall,
        graph: Optional[Graph] = None,
    ) -> tuple[ToolCall, str]:
        """Execute a tool call and return the result."""
        tool = await resolve_tool_by_name(tool_call.name, context.user_id)
        log.debug(
            f"Executing tool {tool_call.name} (id={tool_call.id}) with args: {tool_call.args}"
        )

        # Send tool call to client
        # await self.send_message(
        #     ToolCallUpdate(
        #         name=tool_call.name,
        #         args=tool_call.args,
        #         message=tool.user_message(tool_call.args),
        #     ).model_dump()
        # )

        result = await tool.process(context, tool_call.args)
        log.debug(f"Tool {tool_call.name} returned: {result}")

        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=tool_call.args,
            result=result,
        ), tool.user_message(tool_call.args)

    def _recursively_model_dump(self, obj):
        """Recursively convert BaseModel instances to dictionaries."""
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: self._recursively_model_dump(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._recursively_model_dump(item) for item in obj]
        else:
            return obj

    def _format_connection_error(self, e: httpx.ConnectError) -> str:
        """Format connection error message."""
        error_msg = str(e)
        if "nodename nor servname provided" in error_msg:
            return "Connection error: Unable to resolve hostname. Please check your network connection and API endpoint configuration."
        else:
            return f"Connection error: {error_msg}"
