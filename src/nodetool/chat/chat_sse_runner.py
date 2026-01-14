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

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Iterable
from contextlib import suppress
from typing import cast

from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ContentArrayOfContentPart,
)
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
)
from openai.types.chat.chat_completion_message_custom_tool_call_param import (
    ChatCompletionMessageCustomToolCallParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)

from nodetool.chat.base_chat_runner import BaseChatRunner
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    AudioRef,
    ImageRef,
    MessageAudioContent,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
    Provider,
    ToolCall,
)
from nodetool.metadata.types import (
    Message as ApiMessage,
)
from nodetool.types.workflow import Workflow
from nodetool.workflows.types import Chunk

log = get_logger(__name__)
# Log level is controlled by env (DEBUG/NODETOOL_LOG_LEVEL)


class ChatSSERunner(BaseChatRunner):
    """
    Manages Server-Sent Events connections for chat, handling message processing and streaming.

    This class inherits common chat functionality from BaseChatRunner and implements
    SSE-specific transport methods for one-way server-to-client streaming.

    Unlike WebSocket, SSE only supports server-to-client communication, so the client
    sends a single request with all necessary data, and the server streams back events.
    """

    def __init__(
        self,
        auth_token: str | None = None,
        default_model: str = "gpt-oss:20b",
        default_provider: str = "ollama",
        workflows: list[Workflow] | None = None,
    ):
        super().__init__(auth_token, default_model, default_provider)
        self.message_queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self.is_connected: bool = False
        # Store the provided chat history for this request (used when database is disabled)
        self.provided_history: list[ApiMessage] = []
        self.workflows = workflows or []

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

        # Check if authentication is enforced
        if Environment.enforce_auth():
            if user_id:
                # Upstream gateway already resolved the user; no need to revalidate.
                self.user_id = user_id
                log.debug("Remote auth enabled; using provided user_id without revalidation")
            else:
                # In production or when remote auth is enabled, authentication is required
                if not self.auth_token:
                    raise ValueError("Missing authentication token")

                # Validate token using Supabase
                log.debug("Validating provided auth token")
                is_valid = await self.validate_token(self.auth_token)
                if not is_valid:
                    raise ValueError("Invalid authentication token")
        else:
            # In local development without enforced auth, use provided user_id or default
            self.user_id = user_id or "1"
            log.debug("Skipping authentication in local development mode")

        if not self.user_id:
            # Ensure user_id is set for downstream processing
            self.user_id = "1"

        self.is_connected = True
        log.info("SSE connection established for chat")

    async def disconnect(self) -> None:
        """
        Close the SSE connection and clean up resources.
        """
        # Stop any ongoing task
        if self.current_task and not self.current_task.done():
            log.debug("Stopping current task during disconnect")
            self.current_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.current_task

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

    async def receive_message(self) -> dict | None:
        """
        SSE doesn't support client-to-server messages after initial request.
        This method is not used in SSE context.

        Returns:
            None - SSE is one-way communication
        """
        return None

    async def get_chat_history_from_db(self, thread_id: str) -> list[ApiMessage]:
        return []

    async def handle_message(self, messages: list[ApiMessage]):
        """
        Handle a message for SSE by preparing the message list and calling handle_message_impl.
        """
        try:
            # Call the implementation method with the loaded messages
            await self.handle_message_impl(messages)

        except asyncio.CancelledError:
            log.info("Message processing cancelled by user")
            # Send cancellation message
            with suppress(Exception):
                await self.send_message(
                    {
                        "type": "generation_stopped",
                        "message": "Generation stopped by user",
                    }
                )
        except Exception as e:
            log.error(f"Error processing message: {str(e)}", exc_info=True)
            error_message = {"type": "error", "message": str(e)}
            with suppress(Exception):
                await self.send_message(error_message)

    def _validate_openai_request(self, request_data: dict) -> dict:
        """
        Validate the OpenAI Chat Completions API request format.

        Args:
            request_data: Raw request data

        Returns:
            Validated request data as dict with proper OpenAI format

        Raises:
            ValidationError: If the request doesn't match OpenAI API format
        """
        try:
            # Ensure required fields for validation
            if "messages" not in request_data:
                raise ValueError("Chat Completions API requires 'messages' field")

            # Create a copy to avoid modifying the original
            validation_data = request_data.copy()
            if "model" not in validation_data:
                raise ValueError("Chat Completions API requires 'model' field")

            # Basic validation of messages format
            messages = validation_data["messages"]
            if not isinstance(messages, list) or len(messages) == 0:
                raise ValueError("Messages must be a non-empty list")

            return validation_data
        except (KeyError, TypeError, AttributeError) as e:
            log.error(f"Invalid Chat Completions API request: {e}")
            raise ValueError(f"Request validation failed: {e}") from e

    def _convert_openai_content(
        self,
        content: str | Iterable[ChatCompletionContentPartParam] | Iterable[ContentArrayOfContentPart] | None,
    ) -> list[MessageContent]:
        """
        Convert OpenAI message content to internal MessageContent list format.

        Args:
            content: OpenAI content (string or array of content parts)

        Returns:
            List of internal MessageContent objects
        """
        if isinstance(content, str):
            # Simple text content
            return [MessageTextContent(text=content)]
        elif isinstance(content, list):
            # Multi-modal content
            message_contents: list[MessageContent] = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        message_contents.append(MessageTextContent(text=part.get("text", "")))
                    elif part_type == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                        message_contents.append(MessageImageContent(image=ImageRef(uri=url)))
                    elif part_type == "input_audio":
                        # Handle audio input
                        message_contents.append(MessageAudioContent(audio=AudioRef()))
                else:
                    # Handle typed objects from OpenAI SDK
                    if hasattr(part, "type"):
                        if part.type == "text" and hasattr(part, "text"):
                            text_content = cast("str", part.text)
                            message_contents.append(MessageTextContent(text=text_content))
                        elif part.type == "image_url" and hasattr(part, "image_url"):
                            image_url_obj = part.image_url
                            url = (
                                cast("str", image_url_obj.url) if hasattr(image_url_obj, "url") else str(image_url_obj)
                            )
                            message_contents.append(MessageImageContent(image=ImageRef(uri=url)))
                        elif part.type == "input_audio":
                            message_contents.append(MessageAudioContent(audio=AudioRef()))
            return message_contents
        else:
            # Fallback to text
            return [MessageTextContent(text=str(content))]

    def _convert_openai_messages(
        self, openai_messages: list[ChatCompletionMessageParam], model: str
    ) -> list[ApiMessage]:
        """
        Convert OpenAI messages to internal Message format for history.

        Args:
            openai_messages: List of validated OpenAI message objects

        Returns:
            List of internal ApiMessage objects
        """
        history = []
        for msg in openai_messages:
            role = msg.get("role", "user")
            content = self._convert_openai_content(msg.get("content", None))

            api_message = ApiMessage(
                role=role,
                model=model,
                provider=Provider(self.default_provider),
                content=content,
                name=msg.get("name"),
                tool_call_id=msg.get("tool_call_id"),
            )

            # Convert tool calls if present
            tool_calls_data = msg.get("tool_calls")
            if tool_calls_data:
                api_message.tool_calls = self._convert_openai_tool_calls(tool_calls_data)

            history.append(api_message)

        return history

    def _convert_openai_tool_calls(
        self,
        openai_tool_calls: Iterable[
            ChatCompletionMessageFunctionToolCallParam | ChatCompletionMessageCustomToolCallParam
        ],
    ) -> list[ToolCall]:
        """
        Convert OpenAI tool calls to internal ToolCall format.

        Args:
            openai_tool_calls: List of OpenAI tool call objects

        Returns:
            List of internal ToolCall objects
        """
        tool_calls = []
        for tc in openai_tool_calls:
            if tc.get("type") == "function":
                function = tc.get("function")
                if isinstance(function, dict):
                    tool_call = ToolCall(
                        id=tc.get("id", ""),
                        name=function.get("name", ""),
                        args=json.loads(function.get("arguments", "{}")),
                    )
                    tool_calls.append(tool_call)
        return tool_calls

    def _convert_openai_tools(self, openai_tools: list[ChatCompletionToolParam]) -> list[str]:
        """
        Convert OpenAI tool definitions to internal tool name list.

        Args:
            openai_tools: List of validated OpenAI tool definition objects

        Returns:
            List of tool names
        """
        tool_names = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                function = tool.get("function")
                if isinstance(function, dict):
                    tool_names.append(function.get("name", ""))
        return tool_names

    def _convert_internal_to_openai_chunk(self, chunk: Chunk, model: str) -> ChatCompletionChunk:
        """
        Convert internal chunk format to OpenAI streaming chunk format.

        Args:
            chunk: Internal chunk format
            model: Model name

        Returns:
            Properly typed OpenAI ChatCompletionChunk object
        """
        # Generate a completion ID and timestamp
        completion_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        # Return properly typed ChatCompletionChunk
        return ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=chunk.content),
                    finish_reason=None,
                )
            ],
        )

    def _create_openai_error_chunk(self, error_message: str, model: str) -> ChatCompletionChunk:
        """
        Create an OpenAI format error chunk.

        Args:
            error_message: Error message string

        Returns:
            Properly typed OpenAI ChatCompletionChunk object
        """
        return ChatCompletionChunk(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=f"Error: {error_message}"),
                    finish_reason="stop",
                )
            ],
        )

    async def stream_response(self, request_data: dict) -> AsyncGenerator[str, None]:
        """
        Process a chat request and stream OpenAI Chat Completions API compatible responses.

        This method expects OpenAI Chat Completions API format and always outputs
        OpenAI compatible streaming responses in Server-Sent Events format.

        Args:
            request_data: OpenAI Chat Completions API format:
                {"messages": [...], "model": "...", "stream": true, ...}

        Yields:
            SSE-formatted event strings in OpenAI format
        """
        # Initialize validated_request to None
        validated_request = None

        try:
            # Validate OpenAI request format first
            log.debug("Validating OpenAI Chat Completions API request format")
            validated_request = self._validate_openai_request(request_data)

            # Convert to internal format
            log.debug("Converting validated OpenAI request to internal format")
            messages = self._convert_openai_messages(validated_request["messages"], validated_request["model"])

            # Process the message in a background task using internal format
            self.current_task = asyncio.create_task(self.handle_message(messages))

            # Stream messages from the queue
            while True:
                try:
                    # Wait for messages with a timeout to check if processing is done
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)

                    if message is None:
                        # End of stream
                        break

                    if message.get("type") == "chunk":
                        chunk = Chunk(**message)

                        # Convert internal message to OpenAI chunk format and yield as SSE
                        openai_chunk: ChatCompletionChunk = self._convert_internal_to_openai_chunk(
                            chunk, validated_request["model"]
                        )
                        yield f"data: {openai_chunk.model_dump_json()}\n\n"
                    elif message.get("type") == "error":
                        # Stream provider/internal errors as OpenAI-compatible chunks
                        error_text = str(message.get("message", "Unknown error"))
                        openai_error = self._create_openai_error_chunk(error_text, validated_request["model"])
                        yield f"data: {openai_error.model_dump_json()}\n\n"

                except TimeoutError:
                    # Check if processing task is done
                    if self.current_task.done():
                        # Check for any exceptions
                        try:
                            await self.current_task
                        except Exception as e:
                            # Send error and break
                            model_name = validated_request["model"] if validated_request else "unknown"
                            error_chunk = self._create_openai_error_chunk(str(e), model_name)
                            yield f"data: {error_chunk.model_dump_json()}\n\n"
                        break
                    continue

        except asyncio.CancelledError:
            log.info("SSE streaming cancelled")
            # Send cancellation message in OpenAI format
            model_name = validated_request["model"] if validated_request else "unknown"
            cancel_chunk = self._create_openai_error_chunk("Generation stopped", model_name)
            yield f"data: {cancel_chunk.model_dump_json()}\n\n"
        except Exception as e:
            log.error(f"Error in SSE streaming: {str(e)}", exc_info=True)
            model_name = validated_request["model"] if validated_request else "unknown"
            error_chunk = self._create_openai_error_chunk(str(e), model_name)
            yield f"data: {error_chunk.model_dump_json()}\n\n"
        finally:
            # Send final [DONE] message for OpenAI format
            yield "data: [DONE]\n\n"
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
            # Stream error as OpenAI-compatible chunk
            model = request_data.get("model", self.default_model)
            openai_error = self._create_openai_error_chunk(str(e), model)
            yield f"data: {openai_error.model_dump_json()}\n\n"
            # Do not send DONE explicitly here to match tests expecting a single error event
        finally:
            await self.disconnect()
