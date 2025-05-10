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
import traceback
import uuid
import json
import msgpack
from typing import List, Sequence
from enum import Enum

from fastapi import WebSocket

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
from nodetool.workflows.types import Chunk
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.help import create_help_answer

log = logging.getLogger(__name__)


def provider_from_model(model: str) -> ChatProvider:
    if model.startswith("claude"):
        return get_provider(Provider.Anthropic)
    elif model.startswith("gpt"):
        return get_provider(Provider.OpenAI)
    elif model.startswith("gemini"):
        return get_provider(Provider.Gemini)
    else:
        raise ValueError(f"Unsupported model: {model}")


async def run_tool(
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
    result = await tool.process(context, tool_call.args)

    return ToolCall(
        id=tool_call.id,
        name=tool_call.name,
        args=tool_call.args,
        result=result,
    )


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

    async def connect(self, websocket: WebSocket):
        """
        Accepts and establishes a new WebSocket connection.

        Args:
            websocket (WebSocket): The FastAPI WebSocket object representing the client connection.

        Raises:
            WebSocketDisconnect: If authentication fails.
        """
        # Validate authentication if token is required in your environment
        # This is where you would implement your authentication logic
        if Environment.is_production() and not self.auth_token:
            # Close connection with 401 Unauthorized status code
            await websocket.close(code=1008, reason="Missing authentication")
            log.warning("WebSocket connection rejected: Missing authentication")
            return

        if self.auth_token:
            # Validate token (implementation depends on your auth system)
            is_valid = await self.validate_token(self.auth_token)
            if not is_valid:
                await websocket.close(code=1008, reason="Invalid authentication")
                log.warning("WebSocket connection rejected: Invalid authentication")
                return

        await websocket.accept()
        self.websocket = websocket
        log.info("WebSocket connection established for chat")

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
        return True

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
                    data = msgpack.unpackb(message["bytes"])
                    self.mode = WebSocketMode.BINARY
                    log.debug("Received binary message")
                elif "text" in message:
                    data = json.loads(message["text"])
                    self.mode = WebSocketMode.TEXT
                    log.debug("Received text message")
                else:
                    log.warning("Received message with unknown format")
                    continue

                print(data)
                message = Message(**data)

                # Add the new message to chat history
                self.chat_history.append(message)

                # Process the message through the workflow
                if message.workflow_id:
                    response_message = await self.process_messages_for_workflow()
                else:
                    response_message = await self.process_messages()

                # Create a new message from the result

                # Add the response to chat history
                self.chat_history.append(response_message)

                # Send the response back to the client
                # await self.send_message(response_message.model_dump())

            except Exception as e:
                log.error(f"Error processing message: {str(e)}")
                traceback.print_exc()
                error_message = {"type": "error", "message": str(e)}
                await self.send_message(error_message)
                # Optionally, you can decide whether to break the loop or continue
                # break

    async def _process_help_messages(self, model: str) -> Message:
        """
        Processes messages using the integrated help system.

        Args:
            model (str): The name of the model to use for help.

        Returns:
            Message: An assistant message containing the aggregated help content.
        """
        provider = provider_from_model(model)
        accumulated_content = ""
        async for help_text_chunk in create_help_answer(
            provider=provider,
            messages=self.chat_history,
            model=model,
        ):
            accumulated_content += help_text_chunk
            await self.send_message(
                {"type": "chunk", "content": help_text_chunk, "done": False}
            )

        # Signal the end of the help stream
        await self.send_message({"type": "chunk", "content": "", "done": True})
        return Message(
            role="assistant",
            content=accumulated_content if accumulated_content else None,
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
            return await self._process_help_messages(actual_model_name)
        else:
            # Existing logic for regular messages
            processing_context = ProcessingContext()

            content = ""
            unprocessed_messages = []

            def init_tool(name: str) -> Tool:
                tool_class = get_tool_by_name(name)
                if tool_class:
                    return tool_class()
                else:
                    raise ValueError(f"Tool {name} not found")

            if last_message.tools:
                selected_tools = [init_tool(name) for name in last_message.tools]
            else:
                selected_tools = []

            assert last_message.model, "Model is required"

            provider = provider_from_model(last_message.model)

            # Stream the response chunks
            while True:
                messages_to_send = self.chat_history + unprocessed_messages
                unprocessed_messages = []

                async for chunk in provider.generate_messages(
                    messages=messages_to_send,
                    model=last_message.model,
                    tools=selected_tools,
                ):  # type: ignore
                    if isinstance(chunk, Chunk):
                        content += chunk.content
                        # Send intermediate chunks to client
                        await self.send_message(
                            {
                                "type": "chunk",
                                "content": chunk.content,
                                "done": chunk.done,
                            }
                        )
                    elif isinstance(chunk, ToolCall):
                        # Send tool call to client
                        await self.send_message(
                            {"type": "tool_call", "tool_call": chunk.model_dump()}
                        )

                        # Process the tool call
                        tool_result = await run_tool(
                            processing_context, chunk, selected_tools
                        )
                        # Add tool messages to unprocessed messages
                        unprocessed_messages.append(
                            Message(role="assistant", tool_calls=[chunk])
                        )
                        unprocessed_messages.append(
                            Message(
                                role="tool",
                                tool_call_id=tool_result.id,
                                content=json.dumps(tool_result.result),
                            )
                        )

                        # Send tool result to client
                        await self.send_message(
                            {"type": "tool_result", "result": tool_result.model_dump()}
                        )

                # If no more unprocessed messages, we're done
                if not unprocessed_messages:
                    break

            return Message(
                role="assistant",
                content=content if content else None,
            )

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

        processing_context = ProcessingContext(
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
            await self.send_message(update)
            if update["type"] == "job_update" and update["status"] == "completed":
                result = update["result"]

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
        try:
            if self.mode == WebSocketMode.BINARY:
                packed_message = msgpack.packb(message, use_bin_type=True)
                await self.websocket.send_bytes(packed_message)  # type: ignore
                log.debug(f"Sent binary message")
            else:
                await self.websocket.send_text(json.dumps(message))
                log.debug(f"Sent text message")
        except Exception as e:
            log.error(f"Error sending message: {e}")
