"""
WebSocket-based chat runner for handling real-time chat communications.

This module provides a WebSocket implementation for managing chat sessions, supporting both
binary (MessagePack) and text (JSON) message formats. It handles:

- Real-time bidirectional communication for chat messages
- Tool execution and streaming responses
- Workflow processing with job updates
- Support for various content types (text, images, audio, video)

## Chat Protocol

### Authentication
- Bearer Token: Send token in the WebSocket connection request header:
  ```
  Authorization: Bearer <token>
  ```
- API Key: Include API key in connection query parameters:
  ```
  ws://server/chat?api_key=<your_api_key>
  ```
- Session-based: Establish an authenticated session before WebSocket connection
  by sending credentials to the authentication endpoint.

Authentication errors result in connection closure with appropriate status codes:
- 401: Unauthorized (missing or invalid credentials)
- 403: Forbidden (valid credentials but insufficient permissions)

### Message Format
- Binary: Messages are encoded using MessagePack for efficient binary transmission
- Text: Messages are encoded as JSON strings

### Message Types
- Client to Server:
  - Regular chat messages with optional tool specifications
  - Workflow execution requests with workflow_id and optional graph data

- Server to Client:
  - Content chunks: Partial responses during generation
  - Tool calls: When the model requests a tool execution
  - Tool results: After a tool has been executed
  - Job updates: Status updates during workflow execution
  - Error messages: When exceptions occur during processing

### Content Types
The protocol supports multiple content formats:
- Text content
- Image references (ImageRef)
- Audio references (AudioRef)
- Video references (VideoRef)

The main class ChatWebSocketRunner manages the WebSocket connection lifecycle and message
processing, including:
- Connection management
- Message reception and parsing
- Chat history tracking
- Response streaming
- Tool execution
- Workflow processing

Example:
    runner = ChatWebSocketRunner()
    await runner.run(websocket)
"""

import logging
import uuid
import json
import msgpack
from typing import List, Sequence
from enum import Enum
from nodetool.models.workflow import Workflow
from nodetool.types.graph import Graph
from pydantic import BaseModel

import httpx
from fastapi import WebSocket
from nodetool.common.chroma_client import get_collection
from supabase import create_async_client, AsyncClient

from nodetool.agents.agent import Agent
from nodetool.agents.tools import (
    AddLabelTool,
    ArchiveEmailTool,
    BrowserTool,
    ConvertPDFToMarkdownTool,
    CreateWorkflowTool,
    DownloadFileTool,
    EditWorkflowTool,
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
    create_workflow_tools,
)
from nodetool.agents.tools.help_tools import (
    SearchNodesTool,
    SearchExamplesTool,
)
from nodetool.chat.ollama_service import get_ollama_models
from nodetool.chat.providers import get_provider
from nodetool.agents.tools.base import Tool, get_tool_by_name
from nodetool.chat.providers.base import ChatProvider
from nodetool.common.environment import Environment
from nodetool.metadata.types import (
    AudioRef,
    ImageRef,
    Message,
    MessageAudioContent,
    MessageVideoContent,
    Provider,
    ToolCall,
    VideoRef,
)
from nodetool.metadata.types import MessageImageContent, MessageTextContent
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import (
    Chunk,
    ToolCallUpdate,
    TaskUpdate,
    PlanningUpdate,
    SubTaskResult,
    OutputUpdate,
)
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.graph_planner import GraphPlanner

log = logging.getLogger(__name__)

# Enable debug logging for this module
# log.setLevel(logging.DEBUG)

ollama_models: list[str] = []


async def cached_ollama_models() -> list[str]:
    global ollama_models
    if ollama_models:
        return ollama_models

    models = await get_ollama_models()
    ollama_models = [model.name for model in models]
    return ollama_models


async def provider_from_model(model: str) -> ChatProvider:
    log.debug(f"Selecting provider for model: {model}")
    if model.startswith("claude"):
        provider = get_provider(Provider.Anthropic)
        log.debug(f"Selected Anthropic provider for model: {model}")
        return provider
    elif model.startswith("gpt"):
        provider = get_provider(Provider.OpenAI)
        log.debug(f"Selected OpenAI provider for model: {model}")
        return provider
    elif model.startswith("gemini"):
        provider = get_provider(Provider.Gemini)
        log.debug(f"Selected Gemini provider for model: {model}")
        return provider
    elif model in await cached_ollama_models():
        provider = get_provider(Provider.Ollama)
        log.debug(f"Selected Ollama provider for model: {model}")
        return provider
    elif "/" in model:  # HuggingFace models typically have org/model format
        provider = get_provider(Provider.HuggingFace)
        log.debug(f"Selected HuggingFace provider for model: {model}")
        return provider
    else:
        log.error(f"Unsupported model: {model}")
        raise ValueError(f"Unsupported model: {model}")


class WebSocketMode(str, Enum):
    BINARY = "binary"
    TEXT = "text"


class ChatWebSocketRunner:
    """
    Manages WebSocket connections for chat, handling message processing and tool execution.

    This class is responsible for the entire lifecycle of a chat WebSocket connection,
    including:
    - Accepting and establishing connections.
    - Receiving, parsing, and processing messages in binary (MessagePack) or text (JSON) format.
    - Maintaining chat history.
    - Executing tools requested by the language model.
    - Integrating with workflows for more complex interactions.
    - Streaming responses and tool results back to the client.
    """

    def __init__(self, auth_token: str | None = None):
        self.websocket: WebSocket | None = None
        self.chat_history: List[Message] = []
        self.mode: WebSocketMode = WebSocketMode.BINARY
        self.auth_token = auth_token
        self.user_id: str | None = None
        self.supabase: AsyncClient | None = None
        self.all_tools = (
            []
        )  # Initialize empty list, will be populated after user_id is set

    async def init_supabase(self):
        if self.supabase:
            return
        supabase_url = Environment.get_supabase_url()
        supabase_key = Environment.get_supabase_key()
        if supabase_url and supabase_key:
            self.supabase = await create_async_client(supabase_url, supabase_key)
        else:
            if Environment.is_production():
                log.warning(
                    "Supabase URL or Key not configured in production environment."
                )

    async def connect(self, websocket: WebSocket):
        """
        Accepts and establishes a new WebSocket connection.

        Args:
            websocket (WebSocket): The FastAPI WebSocket object representing the client connection.

        Raises:
            WebSocketDisconnect: If authentication fails.
        """
        log.debug("Initializing WebSocket connection")

        # Check if remote authentication is required
        if Environment.use_remote_auth():
            # In production or when remote auth is enabled, authentication is required
            if not self.auth_token:
                # Close connection with 401 Unauthorized status code
                await websocket.close(code=1008, reason="Missing authentication")
                log.warning("WebSocket connection rejected: Missing authentication")
                return

            # Validate token using Supabase
            log.debug("Validating provided auth token")
            is_valid = await self.validate_token(self.auth_token)
            if not is_valid:
                await websocket.close(code=1008, reason="Invalid authentication")
                log.warning("WebSocket connection rejected: Invalid authentication")
                return
        else:
            # In local development without remote auth, set a default user ID
            self.user_id = "1"
            log.debug("Skipping authentication in local development mode")

        await websocket.accept()
        self.websocket = websocket
        log.info("WebSocket connection established for chat")
        log.debug("WebSocket connection ready")

        # Initialize tools after user_id is set
        self._initialize_tools()

    async def validate_token(self, token: str) -> bool:
        """
        Validates the authentication token.

        Args:
            token (str): The authentication token to validate.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        # Implement your token validation logic here
        # This is a placeholder - replace with your actual validation logic
        # Example: Call your auth service, check JWT validity, etc.

        # For demonstration purposes, we're returning True
        # In a real implementation, you would verify the token against your auth system
        await self.init_supabase()
        assert self.supabase, "Supabase client not initialized"
        try:
            # Validate the token using Supabase
            session = await self.supabase.auth.get_session()  # type: ignore
            if session and session.access_token == token:
                # verify the token
                user_response = await self.supabase.auth.get_user(token)
                if user_response and user_response.user:
                    self.user_id = user_response.user.id
                    log.debug(
                        f"Token validated successfully for user: {user_response.user.id}"
                    )
                    return True
                else:
                    log.warning(f"Token validation failed: {user_response}")
                    return False
            log.warning("Token does not match current session or no active session.")
            return False

        except Exception as e:
            log.error(f"Error during Supabase token validation: {e}")
            return False

    async def disconnect(self):
        """
        Closes the WebSocket connection if it is active.
        """
        if self.websocket:
            await self.websocket.close()
        self.websocket = None
        log.info("WebSocket disconnected for chat")

    async def run(self, websocket: WebSocket):
        """
        Main loop for handling an active WebSocket connection.

        Listens for incoming messages, processes them, and sends responses.
        This method handles message parsing, dispatches to either standard
        message processing or workflow processing, and manages error handling.

        Args:
            websocket (WebSocket): The FastAPI WebSocket object for the connection.
        """
        await self.connect(websocket)

        assert self.websocket is not None, "WebSocket is not connected"

        while True:
            try:
                message = await self.websocket.receive()

                if message["type"] == "websocket.disconnect":
                    log.info("Received websocket disconnect message")
                    break

                if "bytes" in message:
                    raw_bytes = message["bytes"]
                    data = msgpack.unpackb(raw_bytes)
                    self.mode = WebSocketMode.BINARY
                elif "text" in message:
                    raw_text = message["text"]
                    data = json.loads(raw_text)
                    self.mode = WebSocketMode.TEXT
                else:
                    log.warning(f"Received message with unknown format: {message}")
                    continue

                log.debug(f"Creating Message object from data: {data}")
                try:
                    message = Message(**data)
                    log.debug(
                        f"Created message - role: {message.role}, model: {message.model}, tools: {message.tools}, workflow_id: {message.workflow_id}"
                    )
                except Exception as e:
                    log.error(f"Failed to create Message object: {e}")
                    raise

                # Add the new message to chat history
                self.chat_history.append(message)
                log.debug("Added message to chat history")

                # Process the message through the workflow
                if message.workflow_id:
                    log.debug(
                        f"Processing workflow message with workflow_id: {message.workflow_id}"
                    )
                    response_message = await self.process_messages_for_workflow()
                else:
                    log.debug(
                        f"Processing regular chat message with model: {message.model}, agent_mode: {message.agent_mode}, workflow_assistant: {message.workflow_assistant}"
                    )
                    if message.agent_mode:
                        response_message = await self.process_agent_messages()
                    else:
                        response_message = await self.process_messages()

                log.debug(
                    f"Response message created - role: {response_message.role}, content length: {len(str(response_message.content or ''))}"
                )

                # Create a new message from the result

                # Add the response to chat history
                self.chat_history.append(response_message)
                log.debug("Appended response message to history")

                # Send the response back to the client
                # await self.send_message(response_message.model_dump())

            except Exception as e:
                log.error(f"Error processing message: {str(e)}", exc_info=True)
                error_message = {"type": "error", "message": str(e)}
                await self.send_message(error_message)
                # Continue processing instead of breaking
                continue

    async def _query_chroma_collections(
        self, collections: list[str], query_text: str, n_results: int = 5
    ) -> str:
        """Query multiple ChromaDB collections and return concatenated results."""
        if not collections or not query_text:
            return ""

        all_results = []

        for collection_name in collections:
            # Get the collection
            collection = get_collection(name=collection_name)

            # Perform query
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
            )

            if results["documents"] and results["documents"][0]:
                # Format results from this collection
                collection_results = f"\n\n### Results from {collection_name}:\n"
                for doc, metadata in zip(
                    results["documents"][0],
                    (
                        results["metadatas"][0]
                        if results["metadatas"]
                        else [{}] * len(results["documents"][0])
                    ),
                ):
                    # Truncate long documents
                    doc_preview = f"{doc[:200]}..." if len(doc) > 200 else doc
                    collection_results += f"\n- {doc_preview}"
                all_results.append(collection_results)

        if all_results:
            return "\n".join(all_results)
        else:
            return ""

    async def _process_help_messages(self, model: str) -> Message:
        """
        Processes messages using the integrated help system with full tool access.

        Args:
            model (str): The name of the model to use for help.

        Returns:
            Message: An assistant message containing the aggregated help content.
        """
        try:
            provider = await provider_from_model(model)
            log.debug(f"Processing help messages with model: {model}")
            from nodetool.chat.help import SYSTEM_PROMPT

            # Create help tools combined with all available tools including CreateWorkflowTool
            help_tools = [
                SearchNodesTool(),
                SearchExamplesTool(),
                CreateWorkflowTool(),
                EditWorkflowTool(),
            ]

            # Combine help tools with all other available tools
            all_available_tools = help_tools + self.all_tools

            # Create effective messages for provider with help system prompt
            effective_messages = [
                Message(role="system", content=SYSTEM_PROMPT)
            ] + self.chat_history

            processing_context = ProcessingContext(user_id=self.user_id)
            accumulated_content = ""
            unprocessed_messages = []

            # Process messages with tool execution using common infrastructure
            while True:
                messages_to_send = effective_messages + unprocessed_messages
                unprocessed_messages = []

                async for chunk in provider.generate_messages(  # type: ignore
                    messages=messages_to_send,
                    model=model,
                    tools=help_tools,
                ):
                    if isinstance(chunk, Chunk):
                        accumulated_content += chunk.content
                        await self.send_message(
                            {"type": "chunk", "content": chunk.content, "done": False}
                        )
                    elif isinstance(chunk, ToolCall):
                        log.debug(f"Processing help tool call: {chunk.name}")

                        # Process the tool call using the common tool execution method
                        # This ensures CreateWorkflowTool and all other tools work properly
                        tool_result = await self.run_tool(
                            processing_context, chunk, all_available_tools
                        )
                        log.debug(
                            f"Help tool {chunk.name} execution complete, id={tool_result.id}"
                        )

                        # Add tool messages to unprocessed messages
                        assistant_msg = Message(role="assistant", tool_calls=[chunk])
                        unprocessed_messages.append(assistant_msg)

                        def recursively_model_dump(obj):
                            """Recursively convert BaseModel instances to dictionaries."""
                            if isinstance(obj, BaseModel):
                                return obj.model_dump()
                            elif isinstance(obj, dict):
                                return {
                                    k: recursively_model_dump(v) for k, v in obj.items()
                                }
                            elif isinstance(obj, (list, tuple)):
                                return [recursively_model_dump(item) for item in obj]
                            else:
                                return obj

                        converted_result = recursively_model_dump(tool_result.result)
                        tool_result_json = json.dumps(converted_result)
                        tool_msg = Message(
                            role="tool",
                            tool_call_id=tool_result.id,
                            content=tool_result_json,
                        )
                        unprocessed_messages.append(tool_msg)

                # If no more unprocessed messages, we're done
                if not unprocessed_messages:
                    break

            # Signal the end of the help stream
            await self.send_message({"type": "chunk", "content": "", "done": True})
            return Message(
                role="assistant",
                content=accumulated_content if accumulated_content else None,
            )
        except httpx.ConnectError as e:
            # Extract the error message from the exception
            error_msg = str(e)
            if "nodename nor servname provided" in error_msg:
                error_msg = "Connection error: Unable to resolve hostname. Please check your network connection and API endpoint configuration."
            else:
                error_msg = f"Connection error: {error_msg}"

            log.error(
                f"httpx.ConnectError in _process_help_messages: {e}", exc_info=True
            )

            # Send error message to client
            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "connection_error",
                }
            )

            # Signal the end of the help stream with error
            await self.send_message({"type": "chunk", "content": "", "done": True})

            # Return an error message
            return Message(
                role="assistant",
                content=f"I encountered a connection error while processing the help request: {error_msg}. Please check your network connection and try again.",
            )

    async def _trigger_workflow_creation(self, objective: str) -> dict:
        """
        Triggers workflow creation using the GraphPlanner for tool-initiated requests.

        This method reuses the logic from _process_create_workflow but is designed
        to be called from tool execution rather than direct message processing.

        Args:
            objective (str): The workflow objective/goal

        Returns:
            dict: Result of the workflow creation process
        """
        try:
            # Get the current model from the last message in chat history
            if not self.chat_history:
                raise ValueError("No chat history available to determine model")

            last_message = self.chat_history[-1]
            if not last_message.model:
                raise ValueError("No model specified in the current conversation")

            provider = await provider_from_model(last_message.model)
            log.debug(f"Triggering workflow creation with model: {last_message.model}")

            planner = GraphPlanner(
                provider=provider,
                model=last_message.model,
                objective=objective,
                verbose=True,
            )

            # Create processing context
            processing_context = ProcessingContext(user_id=self.user_id)

            # Send initial status
            await self.send_message(
                {
                    "type": "planning_update",
                    "phase": "Tool-Initiated Creation",
                    "status": "Starting",
                    "content": f"Creating workflow from tool request: {objective}",
                }
            )

            accumulated_content = ""
            workflow_graph = None

            # Execute graph planning and stream updates
            async for update in planner.create_graph(processing_context):
                if isinstance(update, PlanningUpdate):
                    # Stream planning updates to client
                    await self.send_message(
                        {
                            "type": "planning_update",
                            "phase": update.phase,
                            "status": update.status,
                            "content": update.content,
                            "node_id": update.node_id,
                        }
                    )

                    # Accumulate content for the final result
                    if update.content:
                        accumulated_content += (
                            f"[{update.phase}] {update.status}: {update.content}\n"
                        )

                elif isinstance(update, Chunk):
                    # If any chunks are emitted, send them too
                    await self.send_message(
                        {"type": "chunk", "content": update.content, "done": False}
                    )
                    accumulated_content += update.content

            # Get the generated graph
            if planner.graph:
                workflow_graph = planner.graph.model_dump()

                await self.send_message(
                    {
                        "type": "workflow_created",
                        "graph": workflow_graph,
                    }
                )

                return {
                    "success": True,
                    "graph": workflow_graph,
                    "content": accumulated_content
                    + "\n\nWorkflow created successfully!\n",
                }
            else:
                return {
                    "success": False,
                    "error": "No graph was generated",
                    "content": accumulated_content
                    + "\nWorkflow creation completed but no graph was generated.",
                }

        except Exception as e:
            error_msg = f"Error creating workflow: {str(e)}"
            log.error(f"Error in _trigger_workflow_creation: {e}", exc_info=True)

            # Send error message to client
            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "workflow_creation_error",
                }
            )

            return {"success": False, "error": error_msg}

    async def _trigger_workflow_editing(
        self, objective: str, workflow_id: str | None = None, graph: Graph | None = None
    ) -> dict:
        """
        Triggers workflow editing using the GraphPlanner for tool-initiated requests.

        This method loads an existing workflow and uses GraphPlanner to modify it
        based on the provided objective while preserving relevant existing structure.

        Args:
            objective (str): The workflow editing objective/goal
            workflow_id (str | None): ID of the workflow to edit (if None, tries to get from message)
            graph (Graph | None): The graph to edit (if None, tries to get from message)
        Returns:
            dict: Result of the workflow editing process
        """
        try:
            # Get the current model from the last message in chat history
            if not self.chat_history:
                raise ValueError("No chat history available to determine model")

            last_message = self.chat_history[-1]
            if not last_message.model:
                raise ValueError("No model specified in the current conversation")

            provider = await provider_from_model(last_message.model)
            log.debug(f"Triggering workflow editing with model: {last_message.model}")

            planner = GraphPlanner(
                provider=provider,
                model=last_message.model,
                objective=objective,
                existing_graph=graph,
                verbose=True,
            )

            # Create processing context
            processing_context = ProcessingContext(user_id=self.user_id)

            # Send initial status
            await self.send_message(
                {
                    "type": "planning_update",
                    "phase": "Tool-Initiated Editing",
                    "status": "Starting",
                    "content": f"Editing workflow from tool request: {objective}",
                }
            )

            accumulated_content = ""
            workflow_graph = None

            # Execute graph planning and stream updates
            async for update in planner.create_graph(processing_context):
                if isinstance(update, PlanningUpdate):
                    # Stream planning updates to client
                    await self.send_message(
                        {
                            "type": "planning_update",
                            "phase": update.phase,
                            "status": update.status,
                            "content": update.content,
                            "node_id": update.node_id,
                        }
                    )

                    # Accumulate content for the final result
                    if update.content:
                        accumulated_content += (
                            f"[{update.phase}] {update.status}: {update.content}\n"
                        )

                elif isinstance(update, Chunk):
                    # If any chunks are emitted, send them too
                    await self.send_message(
                        {"type": "chunk", "content": update.content, "done": False}
                    )
                    accumulated_content += update.content

            # Get the generated graph
            if planner.graph:
                workflow_graph = planner.graph.model_dump()

                # Send the complete workflow graph
                await self.send_message(
                    {
                        "type": "workflow_updated",
                        "workflow_id": workflow_id,
                        "graph": workflow_graph,
                    }
                )

                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "graph": workflow_graph,
                    "content": accumulated_content
                    + "\n\nWorkflow edited successfully!\n",
                }
            else:
                return {
                    "success": False,
                    "error": "No graph was generated",
                    "content": accumulated_content
                    + "\nWorkflow editing completed but no graph was generated.",
                }

        except Exception as e:
            error_msg = f"Error editing workflow: {str(e)}"
            log.error(f"Error in _trigger_workflow_editing: {e}", exc_info=True)

            # Send error message to client
            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "workflow_editing_error",
                }
            )

            return {"success": False, "error": error_msg}

    async def process_messages(self) -> Message:
        """
        Process messages without a workflow, typically for general chat interactions.

        This method takes the last message from the chat history, initializes
        any specified tools, and uses a chat provider to generate a response.
        It supports streaming responses and tool execution.
        It also handles requests for the integrated help system.

        Returns:
            Message: The assistant's response message.

        Raises:
            ValueError: If a specified tool is not found or if the model is not specified in the last message.
        """
        last_message = self.chat_history[-1]

        # Check for help request
        if last_message.help_mode:
            log.debug(f"Processing help request with model: {last_message.model}")
            assert last_message.model, "Model is required"
            return await self._process_help_messages(last_message.model)
        # Check for workflow assistant mode
        else:
            # Existing logic for regular messages
            processing_context = ProcessingContext(user_id=self.user_id)

            # Extract query text from the last message for collection search
            query_text = ""
            if isinstance(last_message.content, str):
                query_text = last_message.content
            elif isinstance(last_message.content, list) and last_message.content:
                for content_item in last_message.content:
                    if isinstance(content_item, MessageTextContent):
                        query_text = content_item.text
                        break

            # Query collections if specified
            collection_context = ""
            if last_message.collections:
                log.debug(f"Querying collections: {last_message.collections}")
                collection_context = await self._query_chroma_collections(
                    last_message.collections, query_text, n_results=5
                )
                if collection_context:
                    log.debug(
                        f"Retrieved collection context: {len(collection_context)} characters"
                    )

            content = ""
            unprocessed_messages = []

            def init_tool(name: str) -> Tool:
                # First check if it's a workflow tool (exact match)
                for tool in self.all_tools:
                    if tool.name == name:
                        return tool

                # If not found, try sanitizing the name and check again (for node tools)
                from nodetool.agents.tools.base import sanitize_node_name

                sanitized_name = sanitize_node_name(name)
                for tool in self.all_tools:
                    if tool.name == sanitized_name:
                        return tool

                # If still not found in all_tools, try to get by name from registry
                tool_class = get_tool_by_name(name)
                if tool_class:
                    return tool_class()

                # Try sanitized name in registry too
                tool_class = get_tool_by_name(sanitized_name)
                if tool_class:
                    return tool_class()

                raise ValueError(f"Tool {name} not found")

            if last_message.tools:
                selected_tools = [init_tool(name) for name in last_message.tools]
                log.debug(
                    f"Initialized tools: {[tool.name for tool in selected_tools]}"
                )
            else:
                selected_tools = []

            assert last_message.model, "Model is required"

            try:
                provider = await provider_from_model(last_message.model)
                log.debug(
                    f"Using provider {provider.__class__.__name__} for model {last_message.model}"
                )
                log.debug(f"Chat history length: {len(self.chat_history)} messages")

                # Stream the response chunks
                while True:
                    messages_to_send = self.chat_history + unprocessed_messages

                    # If we have collection context, add it as a system message before the last user message
                    if collection_context:
                        # Find the last user message index
                        last_user_index = -1
                        for i in range(len(messages_to_send) - 1, -1, -1):
                            if messages_to_send[i].role == "user":
                                last_user_index = i
                                break

                        if last_user_index >= 0:
                            # Insert collection context before the last user message
                            collection_message = Message(
                                role="system",
                                content=f"Context from knowledge base:\n{collection_context}",
                            )
                            messages_to_send = (
                                messages_to_send[:last_user_index]
                                + [collection_message]
                                + messages_to_send[last_user_index:]
                            )
                            # Clear collection_context so we don't add it again in subsequent iterations
                            collection_context = ""

                    unprocessed_messages = []

                    log.debug(
                        f"Calling provider.generate_messages with {len(messages_to_send)} messages"
                    )
                    async for chunk in provider.generate_messages(
                        messages=messages_to_send,
                        model=last_message.model,
                        tools=selected_tools,
                    ):  # type: ignore
                        log.debug(
                            f"Received chunk from provider: type={type(chunk).__name__}"
                        )
                        if isinstance(chunk, Chunk):
                            content += chunk.content
                            await self.send_message(chunk.model_dump())
                        elif isinstance(chunk, ToolCall):
                            log.debug(f"Processing tool call: {chunk.name}")

                            # Process the tool call
                            tool_result = await self.run_tool(
                                processing_context, chunk, selected_tools, last_message.graph
                            )
                            log.debug(
                                f"Tool {chunk.name} execution complete, id={tool_result.id}"
                            )
                            # Add tool messages to unprocessed messages
                            # Note: Assistant message with tool calls typically has no content
                            assistant_msg = Message(
                                role="assistant", tool_calls=[chunk]
                            )
                            log.debug(
                                f"Creating assistant message with tool call, content={assistant_msg.content}"
                            )
                            unprocessed_messages.append(assistant_msg)

                            def recursively_model_dump(obj):
                                """Recursively convert BaseModel instances to dictionaries."""
                                if isinstance(obj, BaseModel):
                                    return obj.model_dump()
                                elif isinstance(obj, dict):
                                    return {
                                        k: recursively_model_dump(v)
                                        for k, v in obj.items()
                                    }
                                elif isinstance(obj, (list, tuple)):
                                    return [
                                        recursively_model_dump(item) for item in obj
                                    ]
                                else:
                                    return obj

                            converted_result = recursively_model_dump(
                                tool_result.result
                            )
                            tool_result_json = json.dumps(converted_result)
                            tool_msg = Message(
                                role="tool",
                                tool_call_id=tool_result.id,
                                content=tool_result_json,
                            )
                            log.debug(
                                f"Creating tool message with result, content_length={len(tool_msg.content or '')}"
                            )
                            unprocessed_messages.append(tool_msg)

                    # If no more unprocessed messages, we're done
                    if not unprocessed_messages:
                        log.debug("No more unprocessed messages, completing generation")
                        break
                    else:
                        log.debug(
                            f"Have {len(unprocessed_messages)} unprocessed messages, continuing loop"
                        )

                return Message(
                    role="assistant",
                    content=content if content else None,
                )
            except httpx.ConnectError as e:
                # Extract the error message from the exception
                error_msg = str(e)
                if "nodename nor servname provided" in error_msg:
                    error_msg = "Connection error: Unable to resolve hostname. Please check your network connection and API endpoint configuration."
                else:
                    error_msg = f"Connection error: {error_msg}"

                log.error(f"httpx.ConnectError in process_messages: {e}", exc_info=True)

                # Send error message to client
                await self.send_message(
                    {
                        "type": "error",
                        "message": error_msg,
                        "error_type": "connection_error",
                    }
                )

                # Return an error message
                return Message(
                    role="assistant",
                    content=f"I encountered a connection error: {error_msg}. Please check your network connection and try again.",
                )
            except Exception:
                # Re-raise other exceptions to be handled by the outer try-catch
                raise

    async def process_agent_messages(self) -> Message:
        """
        Process messages using the Agent, similar to the CLI implementation.

        Returns:
            Message: The assistant's response message after agent execution.
        """
        last_message = self.chat_history[-1]
        assert last_message.model, "Model is required for agent mode"

        # Extract objective from message content
        if isinstance(last_message.content, str):
            objective = last_message.content
        elif isinstance(last_message.content, list) and last_message.content:
            # Find the first text content
            for content_item in last_message.content:
                if isinstance(content_item, MessageTextContent):
                    objective = content_item.text
                    break
            else:
                objective = "Complete the requested task"
        else:
            objective = "Complete the requested task"

        # Ensure tools are initialized
        if not self.all_tools:
            self._initialize_tools()

        # Get selected tools based on message.tools
        selected_tools = []
        if last_message.tools:
            tool_names = set(last_message.tools)
            selected_tools = [
                tool for tool in self.all_tools if tool.name in tool_names
            ]
            log.debug(
                f"Selected tools for agent: {[tool.name for tool in selected_tools]}"
            )

        processing_context = ProcessingContext(user_id=self.user_id)

        try:
            provider = await provider_from_model(last_message.model)
            agent = Agent(
                name="Assistant",
                objective=objective,
                provider=provider,
                model=last_message.model,
                tools=selected_tools,
                enable_analysis_phase=False,
                enable_data_contracts_phase=False,
                verbose=False,  # Disable verbose console output for websocket
            )

            accumulated_content = ""

            async for item in agent.execute(processing_context):
                if isinstance(item, Chunk):
                    accumulated_content += item.content
                    # Stream chunk to client using the same format as regular messages
                    await self.send_message(
                        {
                            "type": "chunk",
                            "content": item.content,
                            "done": item.done if hasattr(item, "done") else False,
                        }
                    )
                elif isinstance(item, ToolCall):
                    # Send tool call update
                    await self.send_message(
                        {
                            "type": "tool_call_update",
                            "name": item.name,
                            "args": item.args,
                            "message": f"Calling {item.name}...",
                        }
                    )
                elif isinstance(item, TaskUpdate):
                    # Send task update
                    await self.send_message(
                        {
                            "type": "task_update",
                            "event": item.event,
                            "task": item.task.model_dump() if item.task else None,
                            "subtask": (
                                item.subtask.model_dump() if item.subtask else None
                            ),
                        }
                    )
                elif isinstance(item, PlanningUpdate):
                    # Send planning update
                    await self.send_message(
                        {
                            "type": "planning_update",
                            "phase": item.phase,
                            "status": item.status,
                            "content": item.content,
                            "node_id": item.node_id,
                        }
                    )
                elif isinstance(item, SubTaskResult) and not item.is_task_result:
                    # Send subtask result
                    await self.send_message(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": str(item.result),
                        }
                    )

            # Signal completion
            await self.send_message({"type": "chunk", "content": "", "done": True})

            return Message(
                role="assistant",
                content=accumulated_content if accumulated_content else None,
            )

        except Exception as e:
            log.error(f"Error in agent execution: {e}", exc_info=True)
            error_msg = f"Agent execution error: {str(e)}"

            # Send error message to client
            await self.send_message(
                {"type": "error", "message": error_msg, "error_type": "agent_error"}
            )

            # Return error message
            return Message(role="assistant", content=error_msg)

    def _initialize_tools(self):
        """Initialize all available tools."""
        # Initialize standard tools
        standard_tools = [
            AddLabelTool(),
            ArchiveEmailTool(),
            BrowserTool(),
            ConvertPDFToMarkdownTool(),
            CreateWorkflowTool(),
            DownloadFileTool(),
            EditWorkflowTool(),
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

        # Initialize workflow tools if user_id is available
        workflow_tools = []
        if self.user_id:
            try:
                workflow_tools = create_workflow_tools(self.user_id, limit=200)
                log.debug(f"Loaded {len(workflow_tools)} workflow tools")
            except Exception as e:
                log.warning(f"Failed to load workflow tools: {e}")

        # Load all node packages to populate NODE_BY_TYPE registry
        try:
            from nodetool.packages.registry import Registry
            from nodetool.metadata.node_metadata import get_node_classes_from_namespace
            import importlib

            registry = Registry()
            packages = registry.list_installed_packages()

            total_loaded = 0
            for package in packages:
                if package.nodes:
                    # Collect unique namespaces from this package
                    namespaces = set()
                    for node_metadata in package.nodes:
                        node_type = node_metadata.node_type
                        # Extract namespace from node_type (e.g., "nodetool.text" from "nodetool.text.Concatenate")
                        namespace_parts = node_type.split(".")[:-1]
                        if (
                            len(namespace_parts) >= 2
                        ):  # Must have at least nodetool.something
                            namespace = ".".join(namespace_parts)
                            namespaces.add(namespace)

                    # Load each unique namespace from this package
                    for namespace in namespaces:
                        try:
                            # Try to import the module directly
                            if namespace.startswith("nodetool.nodes."):
                                module_path = namespace
                            else:
                                module_path = f"nodetool.nodes.{namespace}"

                            importlib.import_module(module_path)
                            total_loaded += 1
                        except ImportError:
                            # Try alternative approach using get_node_classes_from_namespace
                            try:
                                if namespace.startswith("nodetool."):
                                    namespace_suffix = namespace[
                                        9:
                                    ]  # Remove 'nodetool.'
                                    get_node_classes_from_namespace(
                                        f"nodetool.nodes.{namespace_suffix}"
                                    )
                                    total_loaded += 1
                                else:
                                    get_node_classes_from_namespace(
                                        f"nodetool.nodes.{namespace}"
                                    )
                                    total_loaded += 1
                            except Exception:
                                # Silent fail for packages that can't be loaded
                                pass

            log.debug(f"Loaded {len(packages)} packages with node types")

        except Exception as e:
            log.warning(f"Failed to load all packages: {e}")
            # Continue anyway - some nodes may still be available

        # Initialize node tools
        from nodetool.workflows.base_node import NODE_BY_TYPE
        from nodetool.agents.tools.node_tool import NodeTool

        node_tools = []
        for node_type, node_class in NODE_BY_TYPE.items():
            try:
                node_tool = NodeTool(node_class)
                node_tools.append(node_tool)
            except Exception as e:
                log.warning(f"Failed to create node tool for {node_type}: {e}")

        if node_tools:
            log.debug(
                f"Loaded {len(node_tools)} node tools from {len(NODE_BY_TYPE)} available node types"
            )

        # Store all available tools
        self.all_tools = standard_tools + workflow_tools + node_tools
        log.debug(f"Initialized {len(self.all_tools)} total tools")

    async def process_messages_for_workflow(self) -> Message:
        """
        Processes messages that are part of a defined workflow.

        This method retrieves the workflow ID from the last message,
        initializes a WorkflowRunner, and executes the workflow. It streams
        updates (including job status and results) back to the client.

        Returns:
            Message: A message containing the final result of the workflow execution.

        Raises:
            AssertionError: If the workflow ID is not present in the last message.
        """
        job_id = str(uuid.uuid4())
        last_message = self.chat_history[-1]
        assert last_message.workflow_id is not None, "Workflow ID is required"

        workflow_runner = WorkflowRunner(job_id=job_id)
        log.debug(
            f"Initialized WorkflowRunner for workflow {last_message.workflow_id} with job_id {job_id}"
        )

        processing_context = ProcessingContext(
            user_id=self.user_id,
            workflow_id=last_message.workflow_id,
        )

        request = RunJobRequest(
            workflow_id=last_message.workflow_id,
            messages=self.chat_history,
            graph=last_message.graph,
        )

        log.info(f"Running workflow for {last_message.workflow_id}")
        result = {}
        async for update in run_workflow(
            request,
            workflow_runner,
            processing_context,
        ):
            await self.send_message(update.model_dump())
            log.debug(f"Workflow update sent: {update.type}")
            if isinstance(update, OutputUpdate):
                result[update.node_name] = update.value

        return self.create_response_message(result)

    def create_response_message(self, result: dict) -> Message:
        """
        Constructs a response Message object from a dictionary of results.

        The method iterates through the result dictionary and converts string values
        to MessageTextContent. For dictionary values, it attempts to interpret them
        as ImageRef, VideoRef, or AudioRef based on a 'type' field.

        Args:
            result (dict): A dictionary containing the data to be included in the message content.
                           Keys are typically identifiers, and values can be strings or dictionaries
                           representing rich media types.

        Returns:
            Message: An assistant Message object populated with content derived from the result.

        Raises:
            ValueError: If an unknown content type is encountered in the result dictionary.
        """
        content = []
        for key, value in result.items():
            if isinstance(value, str):
                content.append(MessageTextContent(text=value))
            elif isinstance(value, list):
                content.append(MessageTextContent(text=" ".join(value)))
            elif isinstance(value, dict):
                if value.get("type") == "image":
                    content.append(MessageImageContent(image=ImageRef(**value)))
                elif value.get("type") == "video":
                    content.append(MessageVideoContent(video=VideoRef(**value)))
                elif value.get("type") == "audio":
                    content.append(MessageAudioContent(audio=AudioRef(**value)))
                else:
                    raise ValueError(f"Unknown type: {value}")
            else:
                raise ValueError(f"Unknown type: {type(value)} {value}")
        return Message(
            role="assistant",
            content=content,
            # You might want to set other fields like id, thread_id, etc.
        )

    async def send_message(self, message: dict):
        """
        Sends a message to the connected WebSocket client.

        The message is encoded in binary (MessagePack) or text (JSON) format
        based on the established mode for the connection.

        Args:
            message (dict): The message payload to send.

        Raises:
            AssertionError: If the WebSocket is not connected.
            Exception: If an error occurs during message sending.
        """
        assert self.websocket, "WebSocket is not connected"
        log.debug(
            f"Sending message type: {message.get('type', 'unknown')}, mode: {self.mode}"
        )
        try:
            if self.mode == WebSocketMode.BINARY:
                packed_message = msgpack.packb(message, use_bin_type=True)
                assert packed_message is not None, "Packed message is None"
                await self.websocket.send_bytes(packed_message)  # type: ignore
                log.debug(
                    f"Sent binary message ({len(packed_message)} bytes): type={message.get('type')}"
                )
            else:
                json_text = json.dumps(message)
                await self.websocket.send_text(json_text)
                log.debug(
                    f"Sent text message ({len(json_text)} chars): type={message.get('type')}"
                )
        except Exception as e:
            log.error(f"Error sending message: {e}", exc_info=True)

    async def run_tool(
        self,
        context: ProcessingContext,
        tool_call: ToolCall,
        tools: Sequence[Tool],
        graph: Graph | None = None,
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
        log.debug(
            f"Executing tool {tool_call.name} (id={tool_call.id}) with args: {tool_call.args}"
        )
        # Send tool call to client
        await self.send_message(
            ToolCallUpdate(
                name=tool_call.name,
                args=tool_call.args,
                message=tool.user_message(tool_call.args),
            ).model_dump()
        )
        result = await tool.process(context, tool_call.args)
        log.debug(f"Tool {tool_call.name} returned: {result}")

        # Special handling for CreateWorkflowTool
        if tool_call.name == "create_workflow" and isinstance(result, dict):
            if result.get("action") == "create_workflow" and result.get("success"):
                objective = result.get("objective", "")
                log.info(
                    f"CreateWorkflowTool triggered workflow creation for: {objective}"
                )

                # Trigger the actual workflow creation process
                try:
                    workflow_result = await self._trigger_workflow_creation(objective)
                    # Update the result to include workflow creation outcome
                    result["workflow_creation_result"] = workflow_result
                    result["message"] = f"Workflow creation completed for: {objective}"
                except Exception as e:
                    log.error(
                        f"Error in workflow creation triggered by tool: {e}",
                        exc_info=True,
                    )
                    result["error"] = f"Workflow creation failed: {str(e)}"
                    result["success"] = False

        # Special handling for EditWorkflowTool
        elif tool_call.name == "edit_workflow" and isinstance(result, dict):
            if result.get("action") == "edit_workflow" and result.get("success"):
                objective = result.get("objective", "")
                workflow_id = result.get("workflow_id")
                log.info(
                    f"EditWorkflowTool triggered workflow editing for: {objective}"
                )

                # Trigger the actual workflow editing process
                try:
                    workflow_result = await self._trigger_workflow_editing(
                        objective, workflow_id, graph
                    )
                    # Update the result to include workflow editing outcome
                    result["workflow_editing_result"] = workflow_result
                    result["message"] = f"Workflow editing completed for: {objective}"
                except Exception as e:
                    log.error(
                        f"Error in workflow editing triggered by tool: {e}",
                        exc_info=True,
                    )
                    result["error"] = f"Workflow editing failed: {str(e)}"
                    result["success"] = False

        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=tool_call.args,
            result=result,
        )
