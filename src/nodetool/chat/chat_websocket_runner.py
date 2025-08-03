"""
WebSocket-based chat runner for handling real-time chat communications.

This module provides a stateless WebSocket implementation for managing chat sessions, supporting both
binary (MessagePack) and text (JSON) message formats. It inherits common functionality from
BaseChatRunner and implements WebSocket-specific transport methods.

The main class ChatWebSocketRunner handles:
- WebSocket connection lifecycle
- Binary (MessagePack) and text (JSON) message encoding/decoding
- Real-time bidirectional communication
- Integration with the base chat runner functionality

Example:
    runner = ChatWebSocketRunner()
    await runner.run(websocket)
"""

import logging
import json
import msgpack
import asyncio
from typing import Optional, Dict, List
from enum import Enum

from fastapi import WebSocket
from nodetool.chat.base_chat_runner import BaseChatRunner
from nodetool.common.environment import Environment
from nodetool.metadata.types import Message as ApiMessage

log = logging.getLogger(__name__)


class WebSocketMode(str, Enum):
    BINARY = "binary"
    TEXT = "text"


class ChatWebSocketRunner(BaseChatRunner):
    """
    Manages WebSocket connections for chat, handling message processing and tool execution.

    This class inherits common chat functionality from BaseChatRunner and implements
    WebSocket-specific transport methods for real-time bidirectional communication.
    """

    def __init__(self, auth_token: str | None = None, use_database: bool = True, default_model: str = "gemma3n:latest", default_provider: str = "ollama"):
        super().__init__(auth_token, use_database, default_model, default_provider)
        self.websocket: WebSocket | None = None
        self.mode: WebSocketMode = WebSocketMode.BINARY
        # In-memory storage for chat history when database is disabled
        self.in_memory_history: Dict[str, List[ApiMessage]] = {}

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

    def _save_message_to_memory(self, thread_id: str, message: ApiMessage) -> None:
        """Save a message to in-memory storage for the thread."""
        if thread_id not in self.in_memory_history:
            self.in_memory_history[thread_id] = []
        self.in_memory_history[thread_id].append(message)
        log.debug(f"Saved message to in-memory storage for thread {thread_id}")

    async def _save_message_to_db_async(self, message_data: dict) -> ApiMessage:
        """
        Override to save to in-memory storage when database is disabled.
        """
        if not self.use_database:
            # Create API message directly and save to memory
            metadata_message = ApiMessage(**message_data)
            thread_id = message_data.get("thread_id", "")
            if thread_id:
                self._save_message_to_memory(thread_id, metadata_message)
            log.debug("Created and saved message to in-memory storage")
            return metadata_message
        
        # Use parent implementation for database mode
        return await super()._save_message_to_db_async(message_data)

    async def get_chat_history_from_db(self, thread_id: str) -> List[ApiMessage]:
        """
        Override to provide in-memory chat history when database is disabled.
        """
        if not self.use_database:
            # Return in-memory history for this thread
            history = self.in_memory_history.get(thread_id, [])
            log.debug(f"Retrieved {len(history)} messages from in-memory storage for thread {thread_id}")
            return history
        
        # Use parent implementation for database mode
        return await super().get_chat_history_from_db(thread_id)

    async def handle_message(self, data: dict):
        """
        Override to save messages to in-memory storage when database is disabled and load messages before calling handle_message_impl.
        """
        try:
            # Extract thread_id from message data and ensure thread exists
            thread_id = data.get("thread_id")
            thread_id = await self.ensure_thread_exists(thread_id)
            
            # Update message data with the thread_id (in case it was created)
            data["thread_id"] = thread_id
            
            # Apply defaults if not specified
            if not data.get("model"):
                data["model"] = self.default_model
            if not data.get("provider"):
                data["provider"] = self.default_provider
            
            # Handle message storage and loading based on database mode
            if self.use_database:
                # Save message to database asynchronously (if enabled)
                try:
                    db_message = await self._save_message_to_db_async(data)
                    # Convert to metadata message for processing
                    metadata_message = self._db_message_to_metadata_message(db_message)
                    log.debug("Saved message to database asynchronously")
                except Exception as db_error:
                    log.error(f"Database save failed, continuing with in-memory message: {db_error}")
                    # Create a fallback API message for processing even if DB save fails
                    metadata_message = ApiMessage(**data)
                    log.debug("Created fallback message for processing")
                
                # Load messages from database
                chat_history = await self.get_chat_history_from_db(thread_id)
            else:
                # Create API message directly and save to memory
                metadata_message = ApiMessage(**data)
                self._save_message_to_memory(thread_id, metadata_message)
                log.debug("Created and saved message to in-memory storage")
                
                # Load messages from in-memory storage
                chat_history = await self.get_chat_history_from_db(thread_id)

            # Call the implementation method with the loaded messages
            await self.handle_message_impl(chat_history)

        except asyncio.CancelledError:
            log.info("Message processing cancelled by user")
            # Send cancellation message
            try:
                await self.send_message({"type": "generation_stopped", "message": "Generation stopped by user"})
            except:
                pass
        except Exception as e:
            log.error(f"Error processing message: {str(e)}", exc_info=True)
            error_message = {"type": "error", "message": str(e)}
            try:
                await self.send_message(error_message)
            except:
                pass

    async def _run_processor(self, processor, chat_history, processing_context, tools, **kwargs):
        """Override to save assistant messages to in-memory storage when database is disabled."""
        if not self.use_database:
            # Custom implementation for in-memory mode
            log.debug(f"Running processor {processor.__class__.__name__} (in-memory mode)")
            
            # Create the processor task
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
                        print(message)
                        if message["type"] == "message":
                            # Save assistant message to in-memory storage
                            try:
                                # Create an ApiMessage from the assistant response
                                thread_id = kwargs.get('thread_id') or chat_history[-1].thread_id if chat_history else ""
                                assistant_message = ApiMessage(
                                    thread_id=thread_id,
                                    role="assistant",
                                    content=message.get("content", ""),
                                    **{k: v for k, v in message.items() if k not in ["type", "content"]}
                                )
                                if thread_id:
                                    self._save_message_to_memory(thread_id, assistant_message)
                                log.debug("Saved assistant message to in-memory storage")
                            except Exception as memory_error:
                                log.error(f"Assistant message memory save failed: {memory_error}")
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
        else:
            # Use parent implementation for database mode
            await super()._run_processor(processor, chat_history, processing_context, tools, **kwargs)

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

    async def receive_message(self) -> Optional[dict]:
        """
        Receive a message from the WebSocket client.
        
        Returns:
            The received message data or None if connection is closed
        """
        assert self.websocket is not None, "WebSocket is not connected"
        
        try:
            message = await self.websocket.receive()
            log.debug(f"Received WebSocket message: {message}")

            if message["type"] == "websocket.disconnect":
                log.info("Received websocket disconnect message")
                return None

            if "bytes" in message:
                raw_bytes = message["bytes"]
                data = msgpack.unpackb(raw_bytes)
                self.mode = WebSocketMode.BINARY
                log.info(f"Received binary WebSocket command: {data}")
                return data
            elif "text" in message:
                raw_text = message["text"]
                data = json.loads(raw_text)
                self.mode = WebSocketMode.TEXT
                log.info(f"Received text WebSocket command: {data}")
                return data
            else:
                log.warning(f"Received message with unknown format: {message}")
                return None
                
        except Exception as e:
            log.error(f"Error receiving message: {str(e)}", exc_info=True)
            raise

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
        while True:
            try:
                data = await self.receive_message()
                
                if data is None:
                    # Connection closed
                    break

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
                self.current_task = asyncio.create_task(self.handle_message(data))

            except asyncio.CancelledError:
                log.info("Message receiving cancelled")
                break
            except Exception as e:
                log.error(f"Error in receive loop: {str(e)}", exc_info=True)
                try:
                    error_message = {"type": "error", "message": str(e)}
                    await self.send_message(error_message)
                except:
                    pass  # Ignore errors when sending error message
                continue