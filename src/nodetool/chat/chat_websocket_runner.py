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
from nodetool.agents.tools.workflow_tool import create_workflow_tools
from nodetool.chat.base_chat_runner import BaseChatRunner
from nodetool.common.environment import Environment
from nodetool.metadata.types import Message as ApiMessage
from nodetool.models.message import Message as DBMessage

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

    def __init__(self, auth_token: str | None = None, default_model: str = "gemma3n:latest", default_provider: str = "ollama"):
        super().__init__(auth_token, default_model, default_provider)
        self.websocket: WebSocket | None = None
        self.mode: WebSocketMode = WebSocketMode.BINARY
        # In-memory storage for chat history when database is disabled
        self.in_memory_history: Dict[str, List[ApiMessage]] = {}

    def _initialize_standard_tools(self):
        from nodetool.agents.tools import (
            AddLabelToEmailTool,
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
            ScreenshotTool,
            SearchEmailTool,
        )
        # Initialize tools after user_id is set
        self.all_tools += [
            AddLabelToEmailTool(),
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
            ScreenshotTool(),
            SearchEmailTool(),
        ]   

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

        self.all_tools = []
        self._initialize_standard_tools()
        self._initialize_node_tools()

        if self.user_id:
            self.all_tools += create_workflow_tools(self.user_id, limit=200)

    def _save_message_to_memory(self, thread_id: str, message: ApiMessage) -> None:
        """Save a message to in-memory storage for the thread."""
        if thread_id not in self.in_memory_history:
            self.in_memory_history[thread_id] = []
        self.in_memory_history[thread_id].append(message)
        log.debug(f"Saved message to in-memory storage for thread {thread_id}")

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
            
            # Save message to database asynchronously (if enabled)
            await self._save_message_to_db_async(data)
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