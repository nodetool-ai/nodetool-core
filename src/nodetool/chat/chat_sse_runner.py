"""
Server-Sent Events (SSE) based chat runner for handling streaming chat communications.

This module provides an SSE implementation for managing chat sessions with one-way
server-to-client streaming. It supports text-based messaging using the SSE protocol
where the client sends a single request and receives a stream of events as the response.

The main class ChatSSERunner handles:
- SSE connection setup and streaming
- Text-based message encoding (JSON)
- One-way server-to-client event streaming
- Integration with the base chat runner functionality

SSE Protocol Details:
- Events are sent as text in the format: "data: {json_payload}\n\n"
- Special events like errors use: "event: error\ndata: {json_payload}\n\n"
- The connection closes after all events are sent

Example:
    runner = ChatSSERunner()
    async for event in runner.stream_response(request_data):
        yield event
"""

import logging
import json
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List

from nodetool.chat.base_chat_runner import BaseChatRunner
from nodetool.metadata.types import Message as ApiMessage
from nodetool.common.environment import Environment

log = logging.getLogger(__name__)


class ChatSSERunner(BaseChatRunner):
    """
    Manages Server-Sent Events connections for chat, handling message processing and streaming.

    This class inherits common chat functionality from BaseChatRunner and implements
    SSE-specific transport methods for one-way server-to-client streaming.
    
    Unlike WebSocket, SSE only supports server-to-client communication, so the client
    sends a single request with all necessary data, and the server streams back events.
    """

    def __init__(self, auth_token: str | None = None, use_database: bool = True):
        super().__init__(auth_token, use_database)
        self.message_queue: asyncio.Queue[Optional[dict]] = asyncio.Queue()
        self.is_connected: bool = False
        # Store the provided chat history for this request
        self.provided_history: List[ApiMessage] = []

    async def connect(self, user_id: str | None = None, **kwargs) -> None:
        """
        Establish an SSE connection by validating authentication.

        Args:
            user_id: Optional user ID for local development
            **kwargs: Additional connection parameters

        Raises:
            ValueError: If authentication fails
        """
        log.debug("Initializing SSE connection")

        # Check if remote authentication is required
        if Environment.use_remote_auth():
            # In production or when remote auth is enabled, authentication is required
            if not self.auth_token:
                raise ValueError("Missing authentication token")

            # Validate token using Supabase
            log.debug("Validating provided auth token")
            is_valid = await self.validate_token(self.auth_token)
            if not is_valid:
                raise ValueError("Invalid authentication token")
        else:
            # In local development without remote auth, use provided user_id or default
            self.user_id = user_id or "1"
            log.debug("Skipping authentication in local development mode")

        self.is_connected = True
        log.info("SSE connection established for chat")

        # Initialize tools after user_id is set
        self._initialize_tools()

    async def disconnect(self) -> None:
        """
        Close the SSE connection and clean up resources.
        """
        # Stop any ongoing task
        if self.current_task and not self.current_task.done():
            log.debug("Stopping current task during disconnect")
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
        
        self.is_connected = False
        # Signal end of stream
        await self.message_queue.put(None)
        log.info("SSE connection closed")

    async def send_message(self, message: dict) -> None:
        """
        Queue a message to be sent via SSE.

        Args:
            message: The message payload to send
        """
        if not self.is_connected:
            log.warning("Attempted to send message on closed SSE connection")
            return
            
        await self.message_queue.put(message)

    async def receive_message(self) -> Optional[dict]:
        """
        SSE doesn't support client-to-server messages after initial request.
        This method is not used in SSE context.
        
        Returns:
            None - SSE is one-way communication
        """
        return None
    
    async def get_chat_history_from_db(self, thread_id: str) -> List[ApiMessage]:
        """
        Override to use provided history when database is disabled.
        """
        if not self.use_database:
            # Use the provided history from the request
            log.debug(f"Using provided history with {len(self.provided_history)} messages")
            return self.provided_history
        
        # Use parent implementation for database mode
        return await super().get_chat_history_from_db(thread_id)

    def format_sse_message(self, message: dict) -> str:
        """
        Format a message according to SSE protocol.

        Args:
            message: The message to format

        Returns:
            SSE-formatted string
        """
        # Determine event type
        event_type = message.get("type", "message")
        
        # Format as SSE
        if event_type in ["error", "end", "generation_stopped"]:
            # Use named events for special types
            return f"event: {event_type}\ndata: {json.dumps(message)}\n\n"
        else:
            # Default data-only format
            return f"data: {json.dumps(message)}\n\n"

    async def stream_response(self, request_data: dict) -> AsyncGenerator[str, None]:
        """
        Process a chat request and stream SSE responses.

        This is the main entry point for SSE chat processing. It takes a single
        request containing all necessary data and streams back events.

        Args:
            request_data: The chat request data containing:
                - thread_id: The conversation thread ID
                - content: The message content
                - model: The AI model to use
                - provider: The provider for the model
                - tools: Optional list of tools to use
                - workflow_id: Optional workflow to execute
                - Other message fields

        Yields:
            SSE-formatted event strings
        """
        try:
            # Extract and store provided history if present
            history_data = request_data.pop("history", None)
            if history_data and not self.use_database:
                # Convert history data to ApiMessage objects
                self.provided_history = [
                    ApiMessage(**msg) if isinstance(msg, dict) else msg
                    for msg in history_data
                ]
                log.debug(f"Using provided history with {len(self.provided_history)} messages")
            
            # Process the message in a background task
            self.current_task = asyncio.create_task(self.handle_message(request_data))
            
            # Stream messages from the queue
            while True:
                try:
                    # Wait for messages with a timeout to check if processing is done
                    message = await asyncio.wait_for(
                        self.message_queue.get(), 
                        timeout=0.1
                    )
                    
                    if message is None:
                        # End of stream
                        break
                        
                    # Format and yield SSE message
                    yield self.format_sse_message(message)
                    
                except asyncio.TimeoutError:
                    # Check if processing task is done
                    if self.current_task.done():
                        # Check for any exceptions
                        try:
                            await self.current_task
                        except Exception as e:
                            # Send error and break
                            error_msg = {"type": "error", "message": str(e)}
                            yield self.format_sse_message(error_msg)
                        break
                    continue
                    
        except asyncio.CancelledError:
            log.info("SSE streaming cancelled")
            # Send cancellation message
            cancel_msg = {"type": "generation_stopped", "message": "Generation stopped"}
            yield self.format_sse_message(cancel_msg)
        except Exception as e:
            log.error(f"Error in SSE streaming: {str(e)}", exc_info=True)
            error_msg = {"type": "error", "message": str(e)}
            yield self.format_sse_message(error_msg)
        finally:
            # Ensure cleanup
            await self.disconnect()

    async def process_single_request(self, request_data: dict) -> AsyncGenerator[str, None]:
        """
        Convenience method to process a single chat request with SSE.

        This method handles the complete lifecycle of an SSE chat request:
        1. Establishes connection with authentication
        2. Processes the message
        3. Streams responses
        4. Cleans up resources

        Args:
            request_data: The complete request data including auth token

        Yields:
            SSE-formatted event strings
        """
        # Extract auth token if provided
        auth_token = request_data.get("auth_token", self.auth_token)
        if auth_token:
            self.auth_token = auth_token
            
        # Remove auth token from request data to avoid saving it
        request_data = request_data.copy()
        request_data.pop("auth_token", None)
        
        try:
            # Connect with authentication
            await self.connect(user_id=request_data.get("user_id"))
            
            # Stream the response
            async for event in self.stream_response(request_data):
                yield event
                
        except Exception as e:
            log.error(f"Error processing SSE request: {str(e)}", exc_info=True)
            error_msg = {"type": "error", "message": str(e)}
            yield self.format_sse_message(error_msg)
        finally:
            await self.disconnect()