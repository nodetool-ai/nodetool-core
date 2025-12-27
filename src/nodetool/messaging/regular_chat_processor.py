"""
Regular Chat Processor Module
==============================

This module provides the RegularChatProcessor for standard conversational
interactions without specialized workflow or agent modes.

Architecture Overview
---------------------

```
┌─────────────────────────────────────────────────────────────────────┐
│                   RegularChatProcessor Flow                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Message                                                        │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  RegularChatProcessor                         │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │                Optional Features                        │ │   │
│  │  │                                                         │ │   │
│  │  │  ┌─────────────────┐    ┌─────────────────────────┐    │ │   │
│  │  │  │ Tool Execution  │    │  Collection Context     │    │ │   │
│  │  │  │ (user-selected  │    │  (RAG retrieval from    │    │ │   │
│  │  │  │  tools)         │    │   ChromaDB)             │    │ │   │
│  │  │  └─────────────────┘    └─────────────────────────┘    │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │              Provider Generation Loop                   │ │   │
│  │  │                                                         │ │   │
│  │  │  while has_tool_calls:                                  │ │   │
│  │  │      ├── Stream Chunk events                            │ │   │
│  │  │      ├── Process ToolCall events                        │ │   │
│  │  │      ├── Execute tool, save results                     │ │   │
│  │  │      └── Continue generation                            │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       │                                                              │
│       ▼                                                              │
│  Streaming Response to Client                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Key Features
------------

1. **Streaming Responses**: Real-time text chunks to client
2. **Tool Execution**: User-selected tools from nodetool registry
3. **Collection Context**: RAG-style retrieval from ChromaDB collections
4. **Asset Handling**: Auto-saves images, audio, video to storage

Tool Execution Flow
-------------------

```
Provider Response
       │
       ▼
┌──────────────┐    Tool    ┌─────────────────────┐
│ Chunk Event  │◄──────────►│ ToolCall Event      │
│ (text)       │            │                     │
└──────┬───────┘            └──────────┬──────────┘
       │                               │
       ▼                               ▼
  Stream to               ┌─────────────────────┐
  Client                  │ Execute Tool        │
                          │ ├── Resolve by name │
                          │ ├── Process args    │
                          │ └── Save assets     │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │ Append to messages, │
                          │ continue generation │
                          └─────────────────────┘
```

Asset Processing
----------------

Tool results may contain binary data (images, audio, video).
The processor automatically:
1. Detects MIME type from magic bytes
2. Hashes content for deduplication
3. Uploads to asset storage
4. Returns URI reference

Module Contents
---------------
- REGULAR_SYSTEM_PROMPT: Default assistant persona
- RegularChatProcessor: Main processor class
- detect_mime_type(): Binary magic byte detection
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import tempfile
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import httpx
from pydantic import BaseModel

from nodetool.agents.tools.tool_registry import resolve_tool_by_name
from nodetool.chat.token_counter import count_json_tokens, get_default_encoding
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    ImageRef,
    Message,
    MessageTextContent,
    ToolCall,
    VideoRef,
)
from nodetool.providers.base import BaseProvider
from nodetool.runtime.resources import require_scope
from nodetool.types.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import (
    Chunk,
)

from .message_processor import MessageProcessor

log = get_logger(__name__)
log.setLevel(logging.DEBUG)


def detect_mime_type(data: bytes) -> str:
    """Detect mime type from bytes magic numbers."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    # Audio
    if (
        data.startswith(b"ID3")
        or data.startswith(b"\xff\xfb")
        or data.startswith(b"\xff\xf3")
        or data.startswith(b"\xff\xf2")
    ):
        return "audio/mpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    # Video
    if len(data) > 12 and (data[4:12] == b"ftypisom" or data[4:12] == b"ftypmp42"):
        return "video/mp4"
    return "application/octet-stream"


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


def _get_encoding_for_model(model: Optional[str]):
    try:
        import tiktoken  # type: ignore

        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except Exception:
                pass
    except Exception:
        pass

    return get_default_encoding()


def _log_tool_definition_token_breakdown(tools: list, model: Optional[str]) -> None:
    if not log.isEnabledFor(logging.DEBUG):
        return

    encoding = _get_encoding_for_model(model)
    per_tool: list[tuple[int, str]] = []
    for tool in tools:
        try:
            per_tool.append((count_json_tokens(tool.tool_param(), encoding=encoding), tool.name))
        except Exception:
            per_tool.append((0, getattr(tool, "name", "unknown")))

    total = sum(tokens for tokens, _ in per_tool)
    per_tool.sort(reverse=True)
    log.debug(
        "Tool definition tokens (model=%s): total=%d tools=%d",
        model,
        total,
        len(per_tool),
    )
    for tokens, name in per_tool[:50]:
        log.debug("  tool=%s tokens=%d", name, tokens)


class RegularChatProcessor(MessageProcessor):
    """
    Standard chat message processor for conversational AI.

    This processor handles regular chat interactions with support for
    tool execution and RAG-style collection queries. It works with
    any LLM provider implementing the BaseProvider interface.

    Processing Flow
    ---------------

    ```
    1. Resolve user-selected tools
    2. Query collections for context (if specified)
    3. Enter generation loop:
       ├── Call provider.generate_messages()
       ├── Stream Chunk events to client
       ├── On ToolCall: execute, append result, continue
       └── Repeat until no more tool calls
    4. Send final completion message
    ```

    Features
    --------
    - **Streaming**: Real-time response chunks via send_message()
    - **Tools**: Execute user-selected tools from registry
    - **Collections**: Query ChromaDB for contextual grounding
    - **Assets**: Auto-save binary results (images, audio, video)
    - **Error Handling**: Graceful connection error recovery

    Example Usage
    -------------
    ```python
    processor = RegularChatProcessor(provider)
    await processor.process(
        chat_history=[user_message],
        processing_context=context,
        collections=["my_knowledge_base"],
    )
    ```

    Attributes
    ----------
    provider : BaseProvider
        The LLM provider for generating responses
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

        if last_message.tools:
            tools = await asyncio.gather(
                *[resolve_tool_by_name(name, processing_context.user_id) for name in last_message.tools]
            )
        else:
            tools = []

        # Extract query text for collection search
        query_text = self._extract_query_text(last_message)

        if chat_history[0].role != "system":
            chat_history = [
                Message(role="system", content=REGULAR_SYSTEM_PROMPT),
                *chat_history,
            ]

        # Query collections if specified
        collection_context = ""
        if collections:
            log.debug(f"Querying collections: {collections}")
            collection_context = await self._query_collections(collections, query_text, n_results=5)
            if collection_context:
                log.debug(f"Retrieved collection context: {len(collection_context)} characters")

        assert last_message.model, "Model is required"

        try:
            # Stream the response chunks
            while True:
                messages_to_send = chat_history + unprocessed_messages

                # Add collection context if available
                if collection_context:
                    messages_to_send = self._add_collection_context(messages_to_send, collection_context)
                    collection_context = ""  # Clear after first use

                unprocessed_messages = []

                _log_tool_definition_token_breakdown(tools, last_message.model)
                log.debug(f"Calling provider.generate_messages with {len(messages_to_send)} messages")
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
                        # Resolve tool and prepare message
                        tool = await resolve_tool_by_name(chunk.name, processing_context.user_id)
                        message_text = tool.user_message(chunk.args)

                        # Create assistant message with tool call
                        assistant_msg = Message(
                            role="assistant",
                            tool_calls=[
                                ToolCall(
                                    id=chunk.id,
                                    name=chunk.name,
                                    args=chunk.args,
                                    result=None,
                                    message=message_text,
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

                        log.debug(f"Processing tool call: {chunk.name}")

                        # Process the tool call
                        # We resolve the tool again inside _run_tool or just execute directly?
                        # Since we already resolved 'tool', we can just run it.
                        # But _run_tool is convenient.
                        # However, _run_tool does logging and message creation which we partly did.
                        # To minimal change, let's keep _run_tool but ignore its returned message or change how we use it.
                        # Actually, keeping _run_tool is fine, it just resolves again (cached?) or cheap.

                        tool_result, _ = await self._run_tool(processing_context, chunk, graph)
                        log.debug(f"Tool {chunk.name} execution complete, id={tool_result.id}")

                        # Convert result to JSON
                        converted_result = await self._process_tool_result(tool_result.result)
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

                    # Log provider call for cost tracking
                    await self._log_provider_call(
                        processing_context.user_id,
                        last_message.provider,
                        last_message.model,
                        processing_context.workflow_id,
                    )

                    break
                else:
                    log.debug(f"Have {len(unprocessed_messages)} unprocessed messages, continuing loop")

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

    async def _query_collections(self, collections: List[str], query_text: str, n_results: int) -> str:
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
            log.debug(f"Query results for {collection_name}: {results}")

            if results["documents"] and results["documents"][0]:
                collection_results = f"\n\n### Results from {collection_name}:\n"
                for doc, _metadata in zip(
                    results["documents"][0],
                    (results["metadatas"][0] if results["metadatas"] else [{}] * len(results["documents"][0])),
                    strict=False,
                ):
                    doc_preview = f"{doc[:200]}..." if len(doc) > 200 else doc
                    collection_results += f"\n- {doc_preview}"
                all_results.append(collection_results)

        return "\n".join(all_results) if all_results else ""

    def _add_collection_context(self, messages: List[Message], collection_context: str) -> List[Message]:
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
            return [
                *messages[:last_user_index],
                collection_message,
                *messages[last_user_index:],
            ]
        return messages

    async def _run_tool(
        self,
        context: ProcessingContext,
        tool_call: ToolCall,
        graph: Optional[Graph] = None,
    ) -> tuple[ToolCall, str]:
        """Execute a tool call and return the result."""
        tool = await resolve_tool_by_name(tool_call.name, context.user_id)
        log.debug(f"Executing tool {tool_call.name} (id={tool_call.id}) with args: {tool_call.args}")

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

    async def _save_asset_ref(self, asset: AssetRef) -> dict:
        """Save data from an AssetRef and return the updated dict."""
        if not asset.data:
            return asset.model_dump()

        data = asset.data
        # If data is a list (e.g. from some conversions), take first element or handle appropriately
        # For now assume data is bytes
        if isinstance(data, list):
            # This case appears in some internal representations
            data = data[0]

        if not isinstance(data, (bytes, bytearray)):
            # If data is not bytes, we can't save it as file easily without more info
            return asset.model_dump()

        # Determine extension based on AssetRef type or sniff if needed
        ext = None
        if isinstance(asset, ImageRef):
            # Try to detect image format
            mime = detect_mime_type(data)
            if mime != "application/octet-stream":
                ext = mimetypes.guess_extension(mime)
            if not ext:
                ext = ".png"  # Default for images
        elif isinstance(asset, AudioRef):
            mime = detect_mime_type(data)
            if mime != "application/octet-stream":
                ext = mimetypes.guess_extension(mime)
            if not ext:
                ext = ".wav"  # Default for audio
        elif isinstance(asset, VideoRef):
            if hasattr(asset, "format") and asset.format:
                ext = f".{asset.format}"
            else:
                mime = detect_mime_type(data)
                if mime != "application/octet-stream":
                    ext = mimetypes.guess_extension(mime)
            if not ext:
                ext = ".mp4"  # Default for video
        else:
            # Generic AssetRef
            mime = detect_mime_type(data)
            ext = mimetypes.guess_extension(mime) or ".bin"

        # Create hash for filename
        sha = hashlib.sha256(data).hexdigest()
        filename = f"{sha}{ext}"

        # Upload to asset storage
        asset_storage = require_scope().get_asset_storage()
        uri = await asset_storage.upload(filename, BytesIO(data))

        # Update the asset with URI and clear data
        asset.uri = uri
        asset.data = None  # Clear data to avoid serialization
        return asset.model_dump()

    async def _save_asset(self, data: bytes) -> dict:
        """Save bytes as an asset and return an AssetRef dict."""
        mime_type = detect_mime_type(data)
        ext = mimetypes.guess_extension(mime_type) or ".bin"

        # Create hash for filename
        sha = hashlib.sha256(data).hexdigest()
        filename = f"{sha}{ext}"

        # Upload to asset storage
        asset_storage = require_scope().get_asset_storage()
        uri = await asset_storage.upload(filename, BytesIO(data))

        if mime_type.startswith("image/"):
            return ImageRef(uri=uri).model_dump()
        elif mime_type.startswith("audio/"):
            return AudioRef(uri=uri).model_dump()
        elif mime_type.startswith("video/"):
            return VideoRef(uri=uri).model_dump()
        else:
            return AssetRef(uri=uri).model_dump()

    async def _process_tool_result(self, obj):
        """Recursively convert BaseModel instances to dictionaries and handle bytes."""
        if isinstance(obj, AssetRef):
            # Special handling for AssetRef and subclasses
            return await self._save_asset_ref(obj)
        elif isinstance(obj, BaseModel):
            # Convert to dict, then recurse
            return await self._process_tool_result(obj.model_dump())
        elif isinstance(obj, dict):
            return {k: await self._process_tool_result(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [await self._process_tool_result(item) for item in obj]
        elif isinstance(obj, (bytes, bytearray)):
            return await self._save_asset(obj)
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()
        else:
            return obj

    def _format_connection_error(self, e: httpx.ConnectError) -> str:
        """Format connection error message."""
        error_msg = str(e)
        if "nodename nor servname provided" in error_msg:
            return "Connection error: Unable to resolve hostname. Please check your network connection and API endpoint configuration."
        else:
            return f"Connection error: {error_msg}"

    async def _log_provider_call(
        self,
        user_id: str,
        provider: str | None,
        model: str | None,
        workflow_id: str,
    ) -> None:
        """
        Log the provider call to the database for cost tracking.

        Args:
            user_id: User ID making the call
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model identifier
            workflow_id: Workflow ID for tracking
        """
        if not provider or not model:
            log.warning("Cannot log provider call: missing provider or model")
            return

        try:
            cost = self.provider.cost

            await self.provider.log_provider_call(
                user_id=user_id,
                provider=str(provider),
                model=model,
                cost=cost,
                workflow_id=workflow_id,
            )
            log.debug(f"Logged provider call: {provider}/{model}, cost={cost}")
        except (KeyError, AttributeError, TypeError) as e:
            # Handle missing or invalid data
            log.warning(f"Failed to log provider call due to invalid data: {e}")
        except Exception as e:
            # Log unexpected errors but don't fail the chat
            log.error(f"Unexpected error logging provider call: {e}", exc_info=True)
