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

import httpx
import chromadb
from fastapi import WebSocket
from nodetool.common.chroma_client import get_chroma_client, get_collection
from supabase import create_async_client, AsyncClient

from nodetool.agents.agent import Agent
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
    create_workflow_tools,
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
from nodetool.workflows.types import Chunk, ToolCallUpdate, TaskUpdate, PlanningUpdate, SubTaskResult, OutputUpdate
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.help import create_help_answer

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
        self.all_tools = []  # Initialize empty list, will be populated after user_id is set

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
                        f"Processing regular chat message with model: {message.model}, agent_mode: {message.agent_mode}"
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

    async def _query_chroma_collections(self, collections: list[str], query_text: str, n_results: int = 5) -> str:
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
                    results["metadatas"][0] if results["metadatas"] else [{}]*len(results["documents"][0])
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
        Processes messages using the integrated help system.

        Args:
            model (str): The name of the model to use for help.

        Returns:
            Message: An assistant message containing the aggregated help content.
        """
        try:
            provider = await provider_from_model(model)
            log.debug(f"Processing help messages with model: {model}")
            accumulated_content = ""
            async for item in create_help_answer(
                provider=provider,
                messages=self.chat_history,
                model=model,
            ):
                if isinstance(item, Chunk):
                    accumulated_content += item.content
                    await self.send_message(
                        {"type": "chunk", "content": item.content, "done": False}
                    )
                elif isinstance(item, ToolCall):
                    await self.send_message(
                        {"type": "tool_call", "tool_call": item.model_dump()}
                    )
                else:
                    log.debug("Help response item was not chunk or tool call")

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
            
            log.error(f"httpx.ConnectError in _process_help_messages: {e}", exc_info=True)
            
            # Send error message to client
            await self.send_message({
                "type": "error",
                "message": error_msg,
                "error_type": "connection_error"
            })
            
            # Signal the end of the help stream with error
            await self.send_message({"type": "chunk", "content": "", "done": True})
            
            # Return an error message
            return Message(
                role="assistant", 
                content=f"I encountered a connection error while processing the help request: {error_msg}. Please check your network connection and try again."
            )

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
        if last_message.model and last_message.model.startswith("help"):
            parts = last_message.model.split(":", 1)
            if len(parts) > 1 and parts[1]:
                actual_model_name = parts[1]
            else:
                actual_model_name = "gpt-4"  # Default model if not specified
            log.debug(f"Processing help request with model: {actual_model_name}")
            return await self._process_help_messages(actual_model_name)
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
                    last_message.collections, 
                    query_text,
                    n_results=5
                )
                if collection_context:
                    log.debug(f"Retrieved collection context: {len(collection_context)} characters")

            content = ""
            unprocessed_messages = []

            def init_tool(name: str) -> Tool:
                # First check if it's a workflow tool
                for tool in self.all_tools:
                    if tool.name == name:
                        return tool
                # If not found in all_tools, try to get by name
                tool_class = get_tool_by_name(name)
                if tool_class:
                    return tool_class()
                else:
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
                                content=f"Context from knowledge base:\n{collection_context}"
                            )
                            messages_to_send = (
                                messages_to_send[:last_user_index] + 
                                [collection_message] + 
                                messages_to_send[last_user_index:]
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
                                processing_context, chunk, selected_tools
                            )
                            log.debug(
                                f"Tool {chunk.name} execution complete, id={tool_result.id}"
                            )
                            # Add tool messages to unprocessed messages
                            # Note: Assistant message with tool calls typically has no content
                            assistant_msg = Message(role="assistant", tool_calls=[chunk])
                            log.debug(
                                f"Creating assistant message with tool call, content={assistant_msg.content}"
                            )
                            unprocessed_messages.append(assistant_msg)

                            tool_msg = Message(
                                role="tool",
                                tool_call_id=tool_result.id,
                                content=json.dumps(tool_result.result),
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
                await self.send_message({
                    "type": "error",
                    "message": error_msg,
                    "error_type": "connection_error"
                })
                
                # Return an error message
                return Message(
                    role="assistant", 
                    content=f"I encountered a connection error: {error_msg}. Please check your network connection and try again."
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
            selected_tools = [tool for tool in self.all_tools if tool.name in tool_names]
            log.debug(f"Selected tools for agent: {[tool.name for tool in selected_tools]}")
        
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
                    await self.send_message({
                        "type": "chunk",
                        "content": item.content,
                        "done": item.done if hasattr(item, 'done') else False
                    })
                elif isinstance(item, ToolCall):
                    # Send tool call update
                    await self.send_message({
                        "type": "tool_call_update",
                        "name": item.name,
                        "args": item.args,
                        "message": f"Calling {item.name}..."
                    })
                elif isinstance(item, TaskUpdate):
                    # Send task update
                    await self.send_message({
                        "type": "task_update",
                        "event": item.event,
                        "task": item.task.model_dump() if item.task else None,
                        "subtask": item.subtask.model_dump() if item.subtask else None
                    })
                elif isinstance(item, PlanningUpdate):
                    # Send planning update
                    await self.send_message({
                        "type": "planning_update",
                        "phase": item.phase,
                        "status": item.status,
                        "content": item.content,
                        "node_id": item.node_id
                    })
                elif isinstance(item, SubTaskResult) and not item.is_task_result:
                    # Send subtask result
                    await self.send_message({
                        "type": "message",
                        "role": "assistant",
                        "content": str(item.result)
                    })
            
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
            await self.send_message({
                "type": "error",
                "message": error_msg,
                "error_type": "agent_error"
            })
            
            # Return error message
            return Message(
                role="assistant",
                content=error_msg
            )

    def _initialize_tools(self):
        """Initialize all available tools."""
        # Initialize standard tools
        standard_tools = [
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
        
        # Initialize workflow tools if user_id is available
        workflow_tools = []
        if self.user_id:
            try:
                workflow_tools = create_workflow_tools(self.user_id, limit=200)
                log.debug(f"Loaded {len(workflow_tools)} workflow tools")
            except Exception as e:
                log.warning(f"Failed to load workflow tools: {e}")
        
        # Store all available tools
        self.all_tools = standard_tools + workflow_tools
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

        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=tool_call.args,
            result=result,
        )
