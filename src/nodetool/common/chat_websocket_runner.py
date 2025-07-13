"""
WebSocket-based chat runner for handling real-time chat communications.

This module provides a stateless WebSocket implementation for managing chat sessions, supporting both
binary (MessagePack) and text (JSON) message formats. It handles:

- Real-time bidirectional communication for chat messages
- Tool execution and streaming responses
- Workflow processing with job updates
- Support for various content types (text, images, audio, video)
- Database persistence of all messages via thread_id

## Database Persistence & Stateless Design

The ChatWebSocketRunner is stateless and fetches chat history from the database on demand:
- Messages are stored in the `nodetool_messages` table
- Each message contains a thread_id for conversation management
- Thread IDs are extracted from incoming messages, not stored in runner state
- Chat history is fetched from the database when needed for processing
- New messages (both user and assistant) are saved to the database asynchronously
- Database operations run in thread pools to avoid blocking WebSocket processing
- No chat history or thread state is kept in memory between processing sessions

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
  - Stop commands: {"type": "stop"} to interrupt ongoing generation

- Server to Client:
  - Content chunks: Partial responses during generation
  - Tool calls: When the model requests a tool execution
  - Tool results: After a tool has been executed
  - Job updates: Status updates during workflow execution
  - Error messages: When exceptions occur during processing
  - Generation stopped: When generation is interrupted by stop command

### Content Types
The protocol supports multiple content formats:
- Text content
- Image references (ImageRef)
- Audio references (AudioRef)
- Video references (VideoRef)

The main class ChatWebSocketRunner manages the WebSocket connection lifecycle and coordinates
message processing using specialized processor classes, including:
- Connection management
- Message reception and parsing
- Database persistence and stateless chat history fetching
- Thread management for conversation continuity
- Delegating to specialized message processors
- Consuming processor message queues and streaming responses

Example:
    runner = ChatWebSocketRunner()
    await runner.run(websocket)
    # thread_id is extracted from each incoming message
"""

import logging
import json
import msgpack
import asyncio
from typing import List, Sequence
from enum import Enum
from nodetool.types.graph import Graph

from fastapi import WebSocket
from nodetool.common.chroma_client import get_collection
from supabase import create_async_client, AsyncClient

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
from nodetool.chat.ollama_service import get_ollama_models
from nodetool.chat.providers import get_provider
from nodetool.agents.tools.base import Tool, get_tool_by_name
from nodetool.common.environment import Environment
from nodetool.models.message import Message as DBMessage
from nodetool.models.thread import Thread
from nodetool.metadata.types import Message as ApiMessage
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.common.message_processors import (
    MessageProcessor,
    RegularChatProcessor,
    HelpMessageProcessor,
    AgentMessageProcessor,
    WorkflowMessageProcessor,
    WorkflowCreationProcessor,
    WorkflowEditingProcessor,
)

log = logging.getLogger(__name__)

# Enable debug logging for this module
log.setLevel(logging.DEBUG)

ollama_models: list[str] = []


async def cached_ollama_models() -> list[str]:
    global ollama_models
    if ollama_models:
        return ollama_models

    models = await get_ollama_models()
    ollama_models = [model.name for model in models]
    return ollama_models


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
    - Fetching chat history from database on demand (stateless design).
    - Executing tools requested by the language model.
    - Integrating with workflows for more complex interactions.
    - Streaming responses and tool results back to the client.
    """

    def __init__(self, auth_token: str | None = None):
        self.websocket: WebSocket | None = None
        self.mode: WebSocketMode = WebSocketMode.BINARY
        self.auth_token = auth_token
        self.user_id: str | None = None
        self.supabase: AsyncClient | None = None
        self.all_tools = (
            []
        )  # Initialize empty list, will be populated after user_id is set
        self.current_task: asyncio.Task | None = None

    def _db_message_to_metadata_message(self, db_message: DBMessage) -> ApiMessage:
        """
        Convert a database Message to a metadata Message.
        
        Args:
            db_message: The database Message to convert
            
        Returns:
            The converted metadata Message
        """
        # Convert graph dict to Graph object if needed
        graph_obj = None
        if db_message.graph:
            try:
                from nodetool.types.graph import Graph
                if isinstance(db_message.graph, dict):
                    graph_obj = Graph(**db_message.graph)
                else:
                    # Already a Graph object or other type
                    graph_obj = db_message.graph
            except Exception as e:
                log.warning(f"Failed to convert graph to Graph object: {e}")
                graph_obj = None
        
        return ApiMessage(
            id=db_message.id,
            workflow_id=db_message.workflow_id,
            graph=graph_obj,
            thread_id=db_message.thread_id,
            tools=db_message.tools,
            tool_call_id=db_message.tool_call_id,
            role=db_message.role or "", 
            name=db_message.name,
            content=db_message.content,
            tool_calls=db_message.tool_calls,
            collections=db_message.collections,
            input_files=db_message.input_files,
            output_files=db_message.output_files,
            created_at=db_message.created_at.isoformat() if db_message.created_at else None,
            provider=db_message.provider,
            model=db_message.model,
            agent_mode=db_message.agent_mode,
            workflow_assistant=db_message.workflow_assistant,
            help_mode=db_message.help_mode,
        )

    def _metadata_message_to_db_message(self, metadata_message: ApiMessage) -> DBMessage:
        """
        Convert a metadata Message to a database Message.
        
        Args:
            metadata_message: The metadata Message to convert
            
        Returns:
            The converted database Message
        """
        # Extract graph as dict if it's a Graph object
        graph_dict = None
        if metadata_message.graph:
            if hasattr(metadata_message.graph, 'model_dump'):
                graph_dict = metadata_message.graph.model_dump()
            elif isinstance(metadata_message.graph, dict):
                graph_dict = metadata_message.graph
        
        return DBMessage.create(
            thread_id=metadata_message.thread_id or "",
            user_id=self.user_id or "",
            workflow_id=metadata_message.workflow_id,
            graph=graph_dict,
            tools=metadata_message.tools,
            tool_call_id=metadata_message.tool_call_id,
            role=metadata_message.role,
            name=metadata_message.name,
            content=metadata_message.content,
            tool_calls=metadata_message.tool_calls,
            collections=metadata_message.collections,
            input_files=metadata_message.input_files,
            output_files=metadata_message.output_files,
            provider=metadata_message.provider,
            model=metadata_message.model,
            agent_mode=metadata_message.agent_mode or False,
            workflow_assistant=metadata_message.workflow_assistant or False,
            help_mode=metadata_message.help_mode or False,
        )

    async def _save_message_to_db_async(self, message_data: dict) -> DBMessage:
        """
        Asynchronously create and save a message to the database.
        
        Args:
            message_data: The message data to save
            
        Returns:
            The created database message
        """
        try:
            # Prepare data for database message creation
            data_copy = message_data.copy()
            data_copy.pop("id", None)  # Remove id if present to avoid conflicts
            data_copy.pop("type", None)  # Remove type field as it's not part of DBMessage
            data_copy.pop("user_id", None)# Remove user_id from data to avoid conflicts since we set it explicitly
            
            # Use thread_id from message data if available
            message_thread_id = data_copy.pop("thread_id", None) or ""
            
            # Run the database operation in a thread pool to avoid blocking
            def _create_db_message():
                return DBMessage.create(
                    thread_id=message_thread_id,
                    user_id=self.user_id or "",
                    **data_copy
                )
            
            # Execute in thread pool to make it non-blocking
            loop = asyncio.get_event_loop()
            db_message = await loop.run_in_executor(None, _create_db_message)
            
            log.debug(f"Saved message {db_message.id} to database asynchronously")
            return db_message
            
        except Exception as e:
            log.error(f"Error saving message to database: {e}", exc_info=True)
            # Re-raise to allow caller to handle gracefully
            raise

    async def get_chat_history_from_db(self, thread_id: str) -> List[ApiMessage]:
        """
        Fetch chat history from the database using thread_id.
        
        Args:
            thread_id: The thread ID to fetch messages for
        
        Returns:
            List[ApiMessage]: The chat history as a list of API messages
        """
        if not thread_id or not self.user_id:
            log.debug("No thread_id or user_id available, returning empty chat history")
            return []
        
        try:
            # Load messages from database using the paginate method
            db_messages, _ = DBMessage.paginate(
                thread_id=thread_id,
                limit=1000,  # Adjust as needed
                reverse=False  # Get messages in chronological order
            )
            
            # Convert database messages to metadata messages
            chat_history = [
                self._db_message_to_metadata_message(db_msg) for db_msg in db_messages
            ]
            log.debug(f"Fetched {len(db_messages)} messages from database for thread {thread_id}")
            return chat_history
        except Exception as e:
            log.error(f"Error fetching chat history from database: {e}", exc_info=True)
            return []



    async def ensure_thread_exists(self, thread_id: str | None = None) -> str:
        """
        Ensure that a thread exists for this conversation.
        Creates a new thread if thread_id is None.
        
        Args:
            thread_id: The thread ID to verify, or None to create a new one
            
        Returns:
            str: The thread ID (existing or newly created)
        """
        if not self.user_id:
            log.warning("Cannot ensure thread: user_id not set")
            raise ValueError("User ID not set")
        
        if not thread_id:
            # Create a new thread
            try:
                thread = Thread.create(user_id=self.user_id)
                log.debug(f"Created new thread {thread.id}")
                return thread.id
            except Exception as e:
                log.error(f"Error creating new thread: {e}", exc_info=True)
                raise
        else:
            # Verify the thread exists and belongs to the user
            try:
                thread = Thread.find(user_id=self.user_id, id=thread_id)
                if not thread:
                    log.warning(f"Thread {thread_id} not found for user {self.user_id}")
                    # Create a new thread as fallback
                    thread = Thread.create(user_id=self.user_id)
                    log.debug(f"Created new thread {thread.id} as fallback")
                    return thread.id
                return thread_id
            except Exception as e:
                log.error(f"Error verifying thread: {e}", exc_info=True)
                raise


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
        # Stop any ongoing threaded event loop
        if self.current_task and not self.current_task.done():
            log.debug("Stopping threaded event loop during disconnect")
            self.current_task.cancel()
        
        if self.websocket:
            await self.websocket.close()
        self.websocket = None
        self.current_task = None
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

        # Create tasks for concurrent message receiving and processing
        receive_task = asyncio.create_task(self._receive_messages())
        
        try:
            # Wait for the receive task to complete (when connection closes)
            await receive_task
        finally:
            # Clean up any running tasks
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()
                try:
                    await self.current_task
                except asyncio.CancelledError:
                    pass

    async def _receive_messages(self):
        """
        Continuously receive messages from the WebSocket and handle them.
        This runs concurrently with message processing to allow stop commands
        to be processed immediately.
        """
        assert self.websocket is not None, "WebSocket is not connected"
        
        while True:
            try:
                message = await self.websocket.receive()
                log.debug(f"Received WebSocket message: {message}")

                if message["type"] == "websocket.disconnect":
                    log.info("Received websocket disconnect message")
                    break

                if "bytes" in message:
                    raw_bytes = message["bytes"]
                    data = msgpack.unpackb(raw_bytes)
                    self.mode = WebSocketMode.BINARY
                    log.info(f"Received binary WebSocket command: {data}")
                elif "text" in message:
                    raw_text = message["text"]
                    data = json.loads(raw_text)
                    self.mode = WebSocketMode.TEXT
                    log.info(f"Received text WebSocket command: {data}")
                else:
                    log.warning(f"Received message with unknown format: {message}")
                    continue

                # Check if this is a stop command
                if isinstance(data, dict) and data.get("type") == "stop":
                    log.debug("Received stop command")
                    if self.current_task and not self.current_task.done():
                        log.debug("Stopping current processor")
                        self.current_task.cancel()
                        await self.send_message({"type": "generation_stopped", "message": "Generation stopped by user"})
                        log.info("Generation stopped by user command")
                    continue

                # If there's already a task running, reject the new message
                if self.current_task and not self.current_task.done():
                    log.debug("Stopping current processor")
                    self.current_task.cancel()

                # Process the message in a background task
                self.current_task = asyncio.create_task(self._handle_message(data))

            except asyncio.CancelledError:
                log.info("Message receiving cancelled")
                break
            except Exception as e:
                log.error(f"Error receiving message: {str(e)}", exc_info=True)
                try:
                    error_message = {"type": "error", "message": str(e)}
                    await self.send_message(error_message)
                except:
                    pass  # Ignore errors when sending error message
                continue

    async def _handle_message(self, data: dict):
        """
        Handle a single message by parsing it and dispatching to the appropriate processor.
        This runs as a background task to avoid blocking message reception.
        """
        try:
            # Extract thread_id from message data and ensure thread exists
            thread_id = data.get("thread_id")
            thread_id = await self.ensure_thread_exists(thread_id)
            
            # Update message data with the thread_id (in case it was created)
            data["thread_id"] = thread_id
            
            # Save message to database asynchronously
            try:
                db_message = await self._save_message_to_db_async(data)
                # Convert to metadata message for processing
                metadata_message = self._db_message_to_metadata_message(db_message)
                log.debug("Saved message to database asynchronously")
            except Exception as db_error:
                log.error(f"Database save failed, continuing with in-memory message: {db_error}")
                # Create a fallback API message for processing even if DB save fails
                from nodetool.metadata.types import Message as ApiMessage
                metadata_message = ApiMessage(**data)
                log.debug("Created fallback message for processing")

            # Process the message through the appropriate processor
            if metadata_message.workflow_id:
                await self.process_messages_for_workflow(thread_id)
            else:
                if metadata_message.agent_mode:
                    await self.process_agent_messages(thread_id)
                else:
                    await self.process_messages(thread_id)

            # Reset processor references after processing
            self.current_task = None

        except asyncio.CancelledError:
            log.info("Message processing cancelled by user")
            # Send cancellation message
            try:
                await self.send_message({"type": "generation_stopped", "message": "Generation stopped by user"})
            except:
                pass  # Ignore errors when sending cancellation message
        except Exception as e:
            log.error(f"Error processing message: {str(e)}", exc_info=True)
            error_message = {"type": "error", "message": str(e)}
            try:
                await self.send_message(error_message)
            except:
                pass  # Ignore errors when sending error message
            # Reset processor references on error
            self.current_task = None

    async def _run_processor(
        self,
        processor: MessageProcessor,
        chat_history: list,
        processing_context: ProcessingContext,
        tools: list,
        **kwargs
    ):
        """Run a processor and stream its messages."""
        log.debug(f"Running processor {processor.__class__.__name__}")
        
        # Create the processor task (don't overwrite current_task as it tracks the overall message handling)
        processor_task = asyncio.create_task(
            processor.process(
                chat_history=chat_history,
                processing_context=processing_context,
                tools=tools,
                **kwargs
            )
        )
        
        try:
            # Process messages while the processor is running
            while processor.has_messages() or processor.is_processing:
                message = await processor.get_message()
                if message:
                    if message["type"] == "message":
                        # Save assistant message to database asynchronously
                        try:
                            await self._save_message_to_db_async(message)
                            log.debug("Saved assistant message to database")
                        except Exception as db_error:
                            log.error(f"Assistant message DB save failed: {db_error}")
                    else:
                        await self.send_message(message)
                else:
                    # Small delay to avoid busy waiting
                    await asyncio.sleep(0.01)
            
            # Wait for the processor task to complete
            await processor_task
            
        except asyncio.CancelledError:
            # If cancelled, make sure the processor task is also cancelled
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
            raise

    async def _trigger_workflow_creation(self, objective: str, thread_id: str):
        """
        Triggers workflow creation using the GraphPlanner for tool-initiated requests.

        This method reuses the logic from _process_create_workflow but is designed
        to be called from tool execution rather than direct message processing.

        Args:
            objective (str): The workflow objective/goal
            thread_id (str): The thread ID to get chat history from

        Returns:
            dict: Result of the workflow creation process
        """
        try:
            # Get the current model from the last message in chat history
            chat_history = await self.get_chat_history_from_db(thread_id)
            if not chat_history:
                raise ValueError("No chat history available to determine model")

            last_message = chat_history[-1]
            if not last_message.provider:
                raise ValueError("No provider specified in the current conversation")
            
            provider = get_provider(last_message.provider)
            processor = WorkflowCreationProcessor(provider, self.user_id)
            processing_context = ProcessingContext(user_id=self.user_id)
            
            await self._run_processor(
                processor=processor,
                chat_history=chat_history,
                processing_context=processing_context,
                tools=self.all_tools,
                objective=objective,
            )
                        
        except Exception as e:
            error_msg = f"Error creating workflow: {str(e)}"
            log.error(f"Error in _trigger_workflow_creation: {e}", exc_info=True)
            return {"success": False, "error": error_msg}

    async def _trigger_workflow_editing(
        self, objective: str, thread_id: str, workflow_id: str | None = None, graph: Graph | None = None
    ):
        """
        Triggers workflow editing using the GraphPlanner for tool-initiated requests.

        This method loads an existing workflow and uses GraphPlanner to modify it
        based on the provided objective while preserving relevant existing structure.

        Args:
            objective (str): The workflow editing objective/goal
            thread_id (str): The thread ID to get chat history from
            workflow_id (str | None): ID of the workflow to edit (if None, tries to get from message)
            graph (Graph | None): The graph to edit (if None, tries to get from message)
        Returns:
            dict: Result of the workflow editing process
        """
        try:
            # Get the current model from the last message in chat history
            chat_history = await self.get_chat_history_from_db(thread_id)
            if not chat_history:
                raise ValueError("No chat history available to determine model")

            last_message = chat_history[-1]
            if not last_message.provider:
                raise ValueError("No provider specified in the current conversation")
            
            provider = get_provider(last_message.provider)
            processor = WorkflowEditingProcessor(provider, self.user_id)
            processing_context = ProcessingContext(user_id=self.user_id)
            
            await self._run_processor(
                processor=processor,
                chat_history=chat_history,
                processing_context=processing_context,
                tools=self.all_tools,
                objective=objective,
                workflow_id=workflow_id,
                graph=graph,
            )
                        
        except Exception as e:
            error_msg = f"Error editing workflow: {str(e)}"
            log.error(f"Error in _trigger_workflow_editing: {e}", exc_info=True)
            return {"success": False, "error": error_msg}

    async def process_messages(self, thread_id: str):
        """
        Process messages without a workflow, typically for general chat interactions.

        This method takes the last message from the chat history, initializes
        any specified tools, and uses a chat provider to generate a response.
        It supports streaming responses and tool execution.
        It also handles requests for the integrated help system.

        Args:
            thread_id (str): The thread ID to get chat history from

        Returns:
            Message: The assistant's response message.

        Raises:
            ValueError: If a specified tool is not found or if the model is not specified in the last message.
        """
        chat_history = await self.get_chat_history_from_db(thread_id)
        if not chat_history:
            raise ValueError("No chat history available")
        
        last_message = chat_history[-1]

        processing_context = ProcessingContext(user_id=self.user_id)
        
        # Check for help request
        if last_message.help_mode:
            log.debug(f"Processing help request with model: {last_message.model}")
            assert last_message.model, "Model is required"
            assert last_message.provider, "Provider is required"
            
            provider = get_provider(last_message.provider)
            processor = HelpMessageProcessor(provider, self.all_tools)
            
            await self._run_processor(
                processor=processor,
                chat_history=chat_history,
                processing_context=processing_context,
                tools=self.all_tools,
            )
        
        # Regular chat processing
        else:

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
                if not last_message.provider:
                    raise ValueError("No provider specified in the current conversation")

                provider = get_provider(last_message.provider)
                log.debug(
                    f"Using provider {provider.__class__.__name__} for model {last_message.model}"
                )
                log.debug(f"Chat history length: {len(chat_history)} messages")
                
                # Create the regular chat processor
                processor = RegularChatProcessor(provider, self.all_tools)
                
                await self._run_processor(
                    processor=processor,
                    chat_history=chat_history,
                    processing_context=processing_context,
                    tools=selected_tools,
                    collections=last_message.collections,
                    graph=last_message.graph,
                )
            except Exception:
                # Re-raise exceptions to be handled by the outer try-catch
                raise

    async def process_agent_messages(self, thread_id: str):
        """
        Process messages using the Agent, similar to the CLI implementation.

        Args:
            thread_id (str): The thread ID to get chat history from

        Returns:
            Message: The assistant's response message after agent execution.
        """
        chat_history = await self.get_chat_history_from_db(thread_id)
        if not chat_history:
            raise ValueError("No chat history available")
        
        last_message = chat_history[-1]
        assert last_message.model, "Model is required for agent mode"
        assert last_message.provider, "Provider is required for agent mode"
        
        # Ensure tools are initialized
        if not self.all_tools:
            self._initialize_tools()
        
        provider = get_provider(last_message.provider)
        processor = AgentMessageProcessor(provider, self.all_tools)
        processing_context = ProcessingContext(user_id=self.user_id)
        
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

        await self._run_processor(
            processor=processor,
            chat_history=chat_history,
            processing_context=processing_context,
            tools=selected_tools,
        )

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

    async def process_messages_for_workflow(self, thread_id: str):
        """
        Processes messages that are part of a defined workflow.

        This method retrieves the workflow ID from the last message,
        initializes a WorkflowRunner, and executes the workflow. It streams
        updates (including job status and results) back to the client.

        Args:
            thread_id (str): The thread ID to get chat history from

        Returns:
            Message: A message containing the final result of the workflow execution.

        Raises:
            AssertionError: If the workflow ID is not present in the last message.
        """
        chat_history = await self.get_chat_history_from_db(thread_id)
        processor = WorkflowMessageProcessor(self.user_id)
        processing_context = ProcessingContext(user_id=self.user_id)
        
        await self._run_processor(
            processor=processor,
            chat_history=chat_history,
            processing_context=processing_context,
            tools=self.all_tools,
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
        try:
            if self.mode == WebSocketMode.BINARY:
                packed_message = msgpack.packb(message, use_bin_type=True)
                assert packed_message is not None, "Packed message is None"
                await self.websocket.send_bytes(packed_message)  # type: ignore
            else:
                json_text = json.dumps(message)
                await self.websocket.send_text(json_text)
        except Exception as e:
            log.error(f"Error sending message: {e}", exc_info=True)