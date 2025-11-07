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

from nodetool.config.logging_config import get_logger
import json
import msgpack
import asyncio
import time
from typing import Optional
from enum import Enum

from fastapi import WebSocket
from fastapi.websockets import WebSocketState
from nodetool.chat.base_chat_runner import BaseChatRunner
from nodetool.config.environment import Environment
from nodetool.runtime.resources import ResourceScope


log = get_logger(__name__)


class WebSocketMode(str, Enum):
    BINARY = "binary"
    TEXT = "text"


class ToolBridge:
    """
    Manages waiting for frontend tool results from WebSocket messages.
    """

    def __init__(self):
        self._futures: dict[str, asyncio.Future] = {}

    def create_waiter(self, tool_call_id: str) -> asyncio.Future:
        """Create a future that will be resolved when tool result arrives."""
        fut = asyncio.get_running_loop().create_future()
        self._futures[tool_call_id] = fut
        return fut

    def resolve_result(self, tool_call_id: str, payload: dict):
        """Resolve the waiting future with the tool result payload."""
        fut = self._futures.pop(tool_call_id, None)
        if fut and not fut.done():
            fut.set_result(payload)

    def cancel_all(self):
        """Cancel all pending tool result waiters."""
        for fut in self._futures.values():
            if not fut.done():
                fut.cancel()
        self._futures.clear()


class ChatWebSocketRunner(BaseChatRunner):
    """
    Manages WebSocket connections for chat, handling message processing and tool execution.

    This class inherits common chat functionality from BaseChatRunner and implements
    WebSocket-specific transport methods for real-time bidirectional communication.
    """

    def __init__(
        self,
        auth_token: str | None = None,
        default_model: str = "gpt-oss:20b",
        default_provider: str = "ollama",
    ):
        super().__init__(auth_token, default_model, default_provider)
        self.websocket: WebSocket | None = None
        self.mode: WebSocketMode = WebSocketMode.BINARY
        # Background heartbeat task to keep the connection alive through proxies
        self.heartbeat_task: asyncio.Task | None = None
        # Frontend tool bridge and manifest
        self.tool_bridge = ToolBridge()
        self.client_tools_manifest: dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, user_id: str | None = None):
        """
        Accepts and establishes a new WebSocket connection.

        Args:
            websocket (WebSocket): The FastAPI WebSocket object representing the client connection.

        Raises:
            WebSocketDisconnect: If authentication fails.
        """
        log.debug("Initializing WebSocket connection")

        # Check if authentication is enforced
        if Environment.enforce_auth():
            if user_id:
                self.user_id = user_id
                log.debug(
                    "Remote auth enabled for WebSocket; using provided user_id without revalidation"
                )
            else:
                # In production or when auth is enforced, authentication is required
                if not self.auth_token:
                    await websocket.close(code=1008, reason="Missing authentication")
                    log.warning("WebSocket connection rejected: Missing authentication")
                    return

                # Validate token using Supabase
                log.debug("Validating provided auth token for WebSocket connection")
                is_valid = await self.validate_token(self.auth_token)
                if not is_valid:
                    await websocket.close(code=1008, reason="Invalid authentication")
                    log.warning("WebSocket connection rejected: Invalid authentication")
                    return
        else:
            # In local development without enforced auth, set a default user ID
            self.user_id = user_id or "1"
            log.debug("Skipping authentication in local development mode")

        if not self.user_id:
            self.user_id = "1"

        await websocket.accept()
        self.websocket = websocket
        log.info("WebSocket connection established for chat")

        # Start heartbeat to keep idle connections alive (skip in tests to avoid leaked tasks)
        if not Environment.is_test():
            if not self.heartbeat_task or self.heartbeat_task.done():
                self.heartbeat_task = asyncio.create_task(self._heartbeat())

    async def handle_message(self, data: dict):
        """
        Handle an incoming WebSocket message by saving to DB and processing using chat history from DB.
        """
        # Wrap database operations in ResourceScope for per-execution isolation
        async with ResourceScope():
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

                # Save message to database asynchronously
                await self._save_message_to_db_async(data)

                # Load history from database
                chat_history = await self.get_chat_history_from_db(thread_id)

                # Call the implementation method with the loaded messages
                await self.handle_message_impl(chat_history)

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

    async def disconnect(self):
        """
        Closes the WebSocket connection if it is active.
        """
        # Stop any ongoing threaded event loop
        if self.current_task and not self.current_task.done():
            log.debug("Stopping threaded event loop during disconnect")
            self.current_task.cancel()

        # Cancel any pending tool result waiters
        self.tool_bridge.cancel_all()

        # Stop heartbeat task
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        self.heartbeat_task = None

        if (
            self.websocket
            and self.websocket.client_state != WebSocketState.DISCONNECTED
        ):
            try:
                await self.websocket.close()
            except Exception as e:
                log.debug(f"WebSocket close ignored: {e}")
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
        if not self.websocket:
            log.debug("Skipping send: WebSocket is not connected")
            return
        # Guard against sending after close
        if (
            getattr(self.websocket, "client_state", None) == WebSocketState.DISCONNECTED
            or getattr(self.websocket, "application_state", None)
            == WebSocketState.DISCONNECTED
        ):
            log.debug("Skipping send: WebSocket is disconnected")
            return
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
                log.debug(f"Received binary WebSocket command: {data}")
                return data
            elif "text" in message:
                raw_text = message["text"]
                data = json.loads(raw_text)
                self.mode = WebSocketMode.TEXT
                log.debug(f"Received text WebSocket command: {data}")
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
        await self.connect(websocket, user_id=self.user_id)

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
            # Ensure we fully disconnect and stop heartbeat
            try:
                await self.disconnect()
            except Exception:
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
                    # Cancel any pending tool result waiters
                    self.tool_bridge.cancel_all()
                    await self.send_message(
                        {
                            "type": "generation_stopped",
                            "message": "Generation stopped by user",
                        }
                    )
                    log.info("Generation stopped by user command")
                    continue

                # Handle client tools manifest
                if (
                    isinstance(data, dict)
                    and data.get("type") == "client_tools_manifest"
                ):
                    tools = data.get("tools", [])
                    self.client_tools_manifest = {tool["name"]: tool for tool in tools}
                    log.debug(f"Received client tools manifest with {len(tools)} tools")
                    continue

                # Handle tool result from frontend
                if isinstance(data, dict) and data.get("type") == "tool_result":
                    tool_call_id = data.get("tool_call_id")
                    if tool_call_id:
                        self.tool_bridge.resolve_result(tool_call_id, data)
                        log.debug(f"Resolved tool result for call_id: {tool_call_id}")
                    continue

                # Respond to ping without invoking processors
                if isinstance(data, dict) and data.get("type") == "ping":
                    await self.send_message({"type": "pong", "ts": time.time()})
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
                except Exception:
                    pass  # Ignore errors when sending error message
                continue

    async def _heartbeat(self):
        """Periodically send a lightweight heartbeat message to keep the WebSocket alive."""
        while True:
            try:
                # Sleep first to avoid sending immediately upon connect
                await asyncio.sleep(25)
                if not self.websocket:
                    break
                await self.send_message({"type": "ping", "ts": time.time()})
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log and continue; transient failures shouldn't kill the heartbeat loop
                log.debug(f"Heartbeat send failed: {e}")
                # On consistent failures, break to avoid spamming after disconnect
                break
