"""
Base class for chat runners providing common functionality for different transport protocols.

This module provides an abstract base class that contains shared logic for handling
chat communications, including:
- Authentication and user management
- Database persistence of messages
- Message processor management
- Tool initialization and execution
- Thread management

Subclasses should implement transport-specific methods for:
- Connection management
- Message sending/receiving
- Protocol-specific formatting
"""

from nodetool.config.logging_config import get_logger
from abc import ABC, abstractmethod
import traceback
from typing import List, Optional
import asyncio

from supabase import create_async_client, AsyncClient

from nodetool.chat.ollama_service import get_ollama_models
from nodetool.chat.providers import get_provider
from nodetool.config.environment import Environment
from nodetool.models.message import Message as DBMessage
from nodetool.models.thread import Thread
from nodetool.metadata.types import Message as ApiMessage, Provider
from nodetool.types.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.messaging.message_processors import (
    MessageProcessor,
    RegularChatProcessor,
    HelpMessageProcessor,
    AgentMessageProcessor,
    WorkflowMessageProcessor,
)

log = get_logger(__name__)

ollama_models: list[str] = []


async def cached_ollama_models() -> list[str]:
    global ollama_models
    if ollama_models:
        return ollama_models

    models = await get_ollama_models()
    ollama_models = [model.name for model in models]
    return ollama_models


class BaseChatRunner(ABC):
    """
    Abstract base class for chat runners providing common functionality.

    This class contains all the shared logic for handling chat interactions,
    including authentication, database operations, message processing, and
    tool management. Subclasses need to implement transport-specific methods.
    """

    def __init__(
        self,
        auth_token: str | None = None,
        default_model: str = "gpt-oss:20b",
        default_provider: str = "ollama",
    ):
        self.auth_token = auth_token
        self.user_id: str | None = None
        self.supabase: AsyncClient | None = None
        self.current_task: asyncio.Task | None = None
        self.default_model = default_model
        self.default_provider = default_provider

    @abstractmethod
    async def connect(self, **kwargs) -> None:
        """
        Establish a connection using the specific transport protocol.
        Subclasses must implement authentication and connection setup.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close the connection and clean up resources.
        """
        pass

    @abstractmethod
    async def send_message(self, message: dict) -> None:
        """
        Send a message using the specific transport protocol.

        Args:
            message: The message payload to send
        """
        pass

    @abstractmethod
    async def receive_message(self) -> Optional[dict]:
        """
        Receive a message from the client.

        Returns:
            The received message data or None if connection is closed
        """
        pass

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
                if isinstance(db_message.graph, dict):
                    graph_obj = Graph(**db_message.graph)
                else:
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
            created_at=(
                db_message.created_at.isoformat() if db_message.created_at else None
            ),
            provider=db_message.provider,
            model=db_message.model,
            agent_mode=db_message.agent_mode,
            workflow_assistant=db_message.workflow_assistant,
            help_mode=db_message.help_mode,
        )

    def _metadata_message_to_db_message(
        self, metadata_message: ApiMessage
    ) -> DBMessage:
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
            if hasattr(metadata_message.graph, "model_dump"):
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
        # Prepare data for database message creation
        data_copy = message_data.copy()
        data_copy.pop("id", None)
        data_copy.pop("type", None)
        data_copy.pop("user_id", None)

        # Normalize tools field to expected types (list[str])
        try:
            tools_value = data_copy.get("tools", None)
            if tools_value is not None:
                normalized_tools: list[str] = []
                if isinstance(tools_value, list):
                    for item in tools_value:
                        if isinstance(item, str):
                            normalized_tools.append(item)
                        elif isinstance(item, dict):
                            name = item.get("name")
                            if isinstance(name, str) and name:
                                normalized_tools.append(name)
                            else:
                                # Fallback to a compact string representation
                                normalized_tools.append(str(item))
                        else:
                            normalized_tools.append(str(item))
                elif isinstance(tools_value, dict):
                    # Convert dict to list of keys if client sent mapping
                    normalized_tools = [str(k) for k in tools_value.keys()]
                else:
                    normalized_tools = [str(tools_value)]
                data_copy["tools"] = normalized_tools
        except Exception:
            # Best-effort normalization; if it fails, drop tools to avoid validation errors
            data_copy["tools"] = None

        # Use thread_id from message data if available
        message_thread_id = data_copy.pop("thread_id", None) or ""

        # Run the database operation in a thread pool to avoid blocking
        # Create database message directly with async
        db_message = await DBMessage.create(
            thread_id=message_thread_id, user_id=self.user_id or "", **data_copy
        )

        log.info(f"Saved message {db_message.id} to database asynchronously")
        return db_message

    async def get_chat_history_from_db(self, thread_id: str) -> List[ApiMessage]:
        """
        Fetch chat history from the database using thread_id.
        When database is disabled, returns empty list (subclasses should override).

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
            db_messages, _ = await DBMessage.paginate(
                thread_id=thread_id, limit=1000, reverse=False
            )

            # Convert database messages to metadata messages
            chat_history = [
                self._db_message_to_metadata_message(db_msg) for db_msg in db_messages
            ]
            log.debug(
                f"Fetched {len(db_messages)} messages from database for thread {thread_id}"
            )
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
                thread = await Thread.create(user_id=self.user_id)
                log.debug(f"Created new thread {thread.id}")
                return thread.id
            except Exception as e:
                log.error(f"Error creating new thread: {e}", exc_info=True)
                raise
        else:
            # Verify the thread exists and belongs to the user
            try:
                thread = await Thread.find(user_id=self.user_id, id=thread_id)
                if not thread:
                    log.warning(f"Thread {thread_id} not found for user {self.user_id}")
                    # Create a new thread as fallback
                    thread = await Thread.create(user_id=self.user_id)
                    log.debug(f"Created new thread {thread.id} as fallback")
                    return thread.id
                return thread_id
            except Exception as e:
                log.error(f"Error verifying thread: {e}", exc_info=True)
                raise

    async def init_supabase(self):
        """Initialize Supabase client if not already initialized."""
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
            session = await self.supabase.auth.get_session()
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

    async def _run_processor(
        self,
        processor: MessageProcessor,
        chat_history: list[ApiMessage],
        processing_context: ProcessingContext,
        **kwargs,
    ):
        """Run a processor and stream its messages."""
        log.debug(f"Running processor {processor.__class__.__name__}")

        async def process():
            try:
                await processor.process(
                    chat_history=chat_history,
                    processing_context=processing_context,
                    **kwargs,
                )
            except Exception as e:
                traceback.print_exc()
                log.error(f"Error during chat processing: {e}")
                await processor.send_message({"type": "error", "message": str(e)})
                processor.is_processing = False

        # Create the processor task
        processor_task = asyncio.create_task(process())
        try:
            # Process messages while the processor is running
            while processor.has_messages() or processor.is_processing:
                message = await processor.get_message()
                if message:
                    if message["type"] == "message":
                        # Persist and forward message events so the client can render them (e.g., tool calls/results)
                        await self._save_message_to_db_async(message)
                        await self.send_message(message)
                        log.debug("Saved and forwarded message to client")
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

    async def process_messages(self, messages: list[ApiMessage]):
        """
        Process messages without a workflow, typically for general chat interactions.
        """
        chat_history = messages
        if not chat_history:
            raise ValueError("No chat history available")

        last_message = chat_history[-1]
        processing_context = ProcessingContext(user_id=self.user_id)

        # Add UI tool support if available
        if hasattr(self, "tool_bridge") and hasattr(self, "client_tools_manifest"):
            processing_context.tool_bridge = self.tool_bridge
            processing_context.ui_tool_names = set(self.client_tools_manifest.keys())
            processing_context.client_tools_manifest = self.client_tools_manifest

        assert last_message.model, "Model is required"

        if not last_message.provider or last_message.provider == Provider.Empty:
            raise ValueError("No provider specified in the current conversation")

        provider = get_provider(last_message.provider)
        log.debug(
            f"Using provider {provider.__class__.__name__} for model {last_message.model}"
        )

        # Check for help request
        if last_message.help_mode:
            log.debug(f"Processing help request with model: {last_message.model}")
            assert last_message.model, "Model is required"
            assert last_message.provider, "Provider is required"

            provider = get_provider(last_message.provider)
            processor = HelpMessageProcessor(provider)

            await self._run_processor(
                processor=processor,
                chat_history=chat_history,
                processing_context=processing_context,
            )

        # Regular chat processing
        else:
            log.debug(f"Chat history length: {len(chat_history)} messages")

            # Create the regular chat processor
            processor = RegularChatProcessor(provider)

            await self._run_processor(
                processor=processor,
                chat_history=chat_history,
                processing_context=processing_context,
                collections=last_message.collections,
                graph=last_message.graph,
            )

    async def process_agent_messages(self, messages: list[ApiMessage]):
        """
        Process messages using the Agent, similar to the CLI implementation.
        """
        chat_history = messages
        if not chat_history:
            raise ValueError("No chat history available")

        last_message = chat_history[-1]
        assert last_message.model, "Model is required for agent mode"
        assert last_message.provider, "Provider is required for agent mode"

        provider = get_provider(last_message.provider)
        processor = AgentMessageProcessor(provider)
        processing_context = ProcessingContext(user_id=self.user_id)

        # Add UI tool support if available
        if hasattr(self, "tool_bridge") and hasattr(self, "client_tools_manifest"):
            processing_context.tool_bridge = self.tool_bridge
            processing_context.ui_tool_names = set(self.client_tools_manifest.keys())
            processing_context.client_tools_manifest = self.client_tools_manifest

        await self._run_processor(
            processor=processor,
            chat_history=chat_history,
            processing_context=processing_context,
        )

    async def process_messages_for_workflow(self, messages: list[ApiMessage]):
        """
        Processes messages that are part of a defined workflow.
        """
        chat_history = messages
        processor = WorkflowMessageProcessor(self.user_id)
        processing_context = ProcessingContext(user_id=self.user_id)

        # Add UI tool support if available
        if hasattr(self, "tool_bridge") and hasattr(self, "client_tools_manifest"):
            processing_context.tool_bridge = self.tool_bridge
            processing_context.ui_tool_names = set(self.client_tools_manifest.keys())
            processing_context.client_tools_manifest = self.client_tools_manifest

        await self._run_processor(
            processor=processor,
            chat_history=chat_history,
            processing_context=processing_context,
        )

    async def handle_message_impl(self, messages: list[ApiMessage]):
        """
        Handle a single message by parsing it and dispatching to the appropriate processor.
        This implementation method should be called by subclasses with the full message list.
        """
        if not messages:
            raise ValueError("No messages provided")

        last_message = messages[-1]
        try:
            # Process the message through the appropriate processor
            if last_message.workflow_id:
                await self.process_messages_for_workflow(messages)
            else:
                if last_message.agent_mode:
                    await self.process_agent_messages(messages)
                else:
                    await self.process_messages(messages)

        except asyncio.CancelledError:
            log.info("Message processing cancelled by user")
            # Send cancellation message
            try:
                await self.send_message(
                    {
                        "type": "generation_stopped",
                        "message": "Generation stopped by user",
                    }
                )
            except Exception:
                pass
        except Exception as e:
            log.error(f"Error processing message: {str(e)}", exc_info=True)
            error_message = {"type": "error", "message": str(e)}
            try:
                await self.send_message(error_message)
            except Exception:
                pass
