"""
Ollama provider implementation for chat completions.

This module implements the ChatProvider interface for Ollama models,
handling message conversion, streaming, and tool integration.
"""

import asyncio
import json
import re
from typing import Any, AsyncGenerator, AsyncIterator, Sequence, Dict
from contextlib import asynccontextmanager

from ollama import AsyncClient, Client
from pydantic import BaseModel
import tiktoken

from nodetool.chat.providers.base import ChatProvider
from nodetool.chat.token_counter import count_messages_tokens
from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.media.image.image_utils import (
    image_data_to_base64_jpeg,
    image_ref_to_base64_jpeg,
)
from nodetool.metadata.types import (
    Message,
    Provider,
    ToolCall,
    MessageImageContent,
    MessageTextContent,
    ImageRef,
)
from nodetool.workflows.types import Chunk

log = get_logger(__name__)


def get_ollama_sync_client() -> Client:
    """Get a sync client for the Ollama API."""
    api_url = Environment.get("OLLAMA_API_URL")
    assert api_url, "OLLAMA_API_URL not set"
    log.debug(f"Creating sync Ollama client with URL: {api_url}")

    return Client(api_url)


@asynccontextmanager
async def get_ollama_client() -> AsyncGenerator[AsyncClient, None]:
    """Get an AsyncClient for the Ollama API."""
    api_url = Environment.get("OLLAMA_API_URL")
    assert api_url, "OLLAMA_API_URL not set"
    log.debug(f"Creating async Ollama client with URL: {api_url}")

    client = AsyncClient(api_url)
    try:
        yield client
    finally:
        log.debug("Closing async Ollama client")
        await client._client.aclose()


class OllamaProvider(ChatProvider):
    """
    Ollama implementation of the ChatProvider interface.

    Handles conversion between internal message format and Ollama's API format,
    as well as streaming completions and tool calling.

    Ollama's message structure follows a specific format:

    1. Message Format:
       - Each message is a dict with "role" and "content" fields
       - Role can be: "user", "assistant", or "tool"
       - Content contains the message text (string)
       - The message history is passed as a list of these message objects

    2. Tool Calls:
       - When a model wants to call a tool, the response includes a "tool_calls" field
       - Each tool call contains:
         - "function": An object with "name" and "arguments" (dict)
         - "arguments" contains the parameters to be passed to the function
       - When responding to a tool call, you provide a message with:
         - "role": "tool"
         - "name": The name of the function that was called
         - "content": The result of the function call

    3. Response Structure:
       - response["message"] contains the model's response
       - It includes fields like "role", "content", and optionally "tool_calls"
       - The response message format is consistent with the input message format
       - If a tool is called, response["message"]["tool_calls"] will be present

    4. Tool Call Flow:
       - Model generates a response with tool_calls
       - Application executes the tool(s) based on arguments
       - Result is sent back as a "tool" role message
       - Model generates a new response incorporating tool results

    For more details, see: https://ollama.com/blog/tool-support

    """

    provider: Provider = Provider.Ollama

    def __init__(self, log_file=None):
        """Initialize the Ollama provider.

        Args:
            log_file (str, optional): Path to a file where API calls and responses will be logged.
                If None, no logging will be performed.
        """
        super().__init__()
        self.api_url = Environment.get("OLLAMA_API_URL")
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.log_file = log_file
        log.debug(
            f"OllamaProvider initialized. API URL present: {bool(self.api_url)}, log_file: {log_file}"
        )

    def get_container_env(self) -> dict[str, str]:
        env_vars = {}
        if self.api_url:
            env_vars["OLLAMA_API_URL"] = self.api_url
        log.debug(f"Container environment variables: {list(env_vars.keys())}")
        return env_vars

    def get_context_length(self, model: str) -> int:
        """Get the maximum token limit for a given model."""
        log.debug(f"Getting context length for model: {model}")
        try:
            client = get_ollama_sync_client()
            log.debug(f"Fetching model info for: {model}")

            # Construct the URL for the show endpoint
            model_info = client.show(model=model).modelinfo
            if model_info is None:
                log.debug("Model info is None, using default context length: 4096")
                return 4096

            log.debug(f"Model info keys: {list(model_info.keys())}")

            for key, value in model_info.items():
                if ".context_length" in key:
                    log.debug(f"Found context length in model info: {key} = {value}")
                    return int(value)

            # Otherwise, try to extract from modelfile parameters
            if model_info["modelfile"]:
                modelfile = model_info["modelfile"]
                param_match = re.search(r"PARAMETER\s+num_ctx\s+(\d+)", modelfile)
                if param_match:
                    context_length = int(param_match.group(1))
                    log.debug(f"Found context length in modelfile: {context_length}")
                    return context_length
                else:
                    log.debug("No num_ctx parameter found in modelfile")

            # Default fallback if we can't determine the context length
            log.debug("Using default context length: 4096")
            return 4096
        except Exception as e:
            log.error(f"Error determining model context length: {e}")
            print(f"Error determining model context length: {e}")
            # Fallback to a reasonable default
            return 4096

    def _count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate token count for a sequence of messages."""
        token_count = count_messages_tokens(messages, encoding=self.encoding)
        log.debug(f"Estimated token count for {len(messages)} messages: {token_count}")
        return token_count

    def convert_message(self, message: Message) -> Dict[str, Any]:
        """
        Convert an internal message to Ollama's format.

        Args:
            message: The message to convert
        """
        log.debug(f"Converting message with role: {message.role}")

        if message.role == "tool":
            log.debug(f"Converting tool message with name: {message.name}")
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            elif isinstance(message.content, dict):
                content = json.dumps(message.content)
            elif isinstance(message.content, list):
                content = json.dumps([part.model_dump() for part in message.content])
            elif isinstance(message.content, str):
                content = message.content
            else:
                content = json.dumps(message.content)
            log.debug(f"Tool message content type: {type(message.content)}")
            return {"role": "tool", "content": content, "name": message.name}
        elif message.role == "system":
            log.debug("Converting system message")
            # Handle system message content conversion
            if message.content is None:
                content = ""
            elif isinstance(message.content, str):
                content = message.content
            else:
                # Handle list content by extracting text from MessageTextContent objects
                text_parts = [
                    part.text
                    for part in message.content
                    if isinstance(part, MessageTextContent)
                ]
                content = "\n".join(text_parts)
                log.debug(f"Extracted {len(text_parts)} text parts from system message")
            return {"role": "system", "content": content}
        elif message.role == "user":
            log.debug("Converting user message")
            assert message.content is not None, "User message content must not be None"
            message_dict: Dict[str, Any] = {"role": "user"}

            if isinstance(message.content, str):
                message_dict["content"] = message.content
                log.debug("User message has string content")
            else:
                # Handle text content
                text_parts = [
                    part.text
                    for part in message.content
                    if isinstance(part, MessageTextContent)
                ]
                message_dict["content"] = "\n".join(text_parts)
                log.debug(f"User message has {len(text_parts)} text parts")

                # Handle image content
                image_parts = [
                    image_ref_to_base64_jpeg(part.image)
                    for part in message.content
                    if isinstance(part, MessageImageContent)
                ]
                if image_parts:
                    message_dict["images"] = image_parts
                    log.debug(f"User message has {len(image_parts)} images")
                else:
                    log.debug("User message has no images")

            return message_dict
        elif message.role == "assistant":
            log.debug("Converting assistant message")
            tool_calls = message.tool_calls or []
            log.debug(f"Assistant message has {len(tool_calls)} tool calls")

            message_dict = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.args,
                        },
                    }
                    for tool_call in tool_calls
                ],
            }

            # Handle None content (common when only tool calls are present)
            if message.content is None:
                message_dict["content"] = ""
                log.debug("Assistant message has no content (tool calls only)")
            elif isinstance(message.content, str):
                message_dict["content"] = message.content
                log.debug("Assistant message has string content")
            else:
                text_parts = [
                    part.text
                    for part in message.content
                    if isinstance(part, MessageTextContent)
                ]
                message_dict["content"] = "\n".join(text_parts)
                log.debug(f"Assistant message has {len(text_parts)} text parts")

            return message_dict
        else:
            log.error(f"Unknown message role: {message.role}")
            raise ValueError(f"Unknown message role {message.role}")

    def format_tools(self, tools: Sequence[Any]) -> list:
        """Convert tools to Ollama's format."""
        log.debug(f"Formatting {len(tools)} tools for Ollama API")
        formatted_tools = [tool.tool_param() for tool in tools]
        log.debug(
            f"Formatted tools: {[tool.get('function', {}).get('name', 'unknown') for tool in formatted_tools]}"
        )
        return formatted_tools

    def _prepare_request_params(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        response_format: dict | None = None,
        max_tokens: int = 4096,
        context_window: int = 4096,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare common parameters for Ollama API requests.

        Args:
            messages: The conversation history
            model: The model to use
            tools: Optional tools to make available to the model
            response_format: Optional response format to pass to the Ollama API

        Returns:
            Dict[str, Any]: Parameters ready for Ollama API request
        """
        log.debug(
            f"Preparing request params for model: {model}, {len(messages)} messages, {len(tools)} tools"
        )

        # Regular message conversion only
        ollama_messages = [self.convert_message(m) for m in messages]
        log.debug(f"Converted to {len(ollama_messages)} Ollama messages")

        params = {
            "model": model,
            "messages": ollama_messages,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": context_window,
            },
        }
        log.debug(f"Set options: num_predict={max_tokens}, num_ctx={context_window}")

        if len(tools) > 0:
            params["tools"] = self.format_tools(tools)
            log.debug(f"Added {len(tools)} tools to request")

        if response_format:
            log.debug(f"Processing response format: {response_format.get('type')}")
            if (
                response_format.get("type") == "json_schema"
                and "json_schema" in response_format
            ):
                schema = response_format["json_schema"]
                if "schema" not in schema:
                    log.error("schema is required in json_schema response format")
                    raise ValueError(
                        "schema is required in json_schema response format"
                    )
                params["format"] = schema["schema"]
                log.debug("Added JSON schema format to request")
            else:
                log.debug("Response format type not supported, skipping")

        log.debug(f"Prepared request params with keys: {list(params.keys())}")
        return params

    def _update_usage_stats(self, response):
        """Update token usage statistics from response."""
        prompt_tokens = getattr(response, "prompt_eval_count", 0)
        completion_tokens = getattr(response, "eval_count", 0)
        # Guard against None from partial/streaming responses
        prompt_tokens = 0 if prompt_tokens is None else int(prompt_tokens)
        completion_tokens = 0 if completion_tokens is None else int(completion_tokens)

        log.debug(
            f"Updating usage stats - prompt: {prompt_tokens}, completion: {completion_tokens}"
        )
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += prompt_tokens + completion_tokens
        log.debug(f"Updated usage stats: {self.usage}")

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """
        Generate streaming completions from Ollama.

        Args:
            messages: The conversation history
            model: The model to use
            tools: Optional tools to make available to the model
            audio: Optional audio
            **kwargs: Additional parameters to pass to the Ollama API

        Yields:
            Chunk | ToolCall: Content chunks or tool calls
        """
        log.debug(f"Starting streaming generation for model: {model}")
        log.debug(f"Streaming with {len(messages)} messages, {len(tools)} tools")
        self._log_api_request(
            "chat_stream", messages, model=model, tools=tools, **kwargs
        )

        async with get_ollama_client() as client:
            params = self._prepare_request_params(
                messages,
                model,
                tools,
                max_tokens=max_tokens,
                context_window=context_window,
                **kwargs,
            )
            params["stream"] = True
            log.debug("Starting streaming chat request")

            completion = await client.chat(**params)
            log.debug("Streaming response initialized")

            chunk_count = 0
            tool_call_count = 0

            async for response in completion:
                chunk_count += 1

                # Track usage metrics when we receive the final response
                if response.done:
                    log.debug("Final chunk received, updating usage stats")
                    self._update_usage_stats(response)

                if response.message.tool_calls is not None:
                    log.debug(
                        f"Chunk contains {len(response.message.tool_calls)} tool calls"
                    )
                    for tool_call in response.message.tool_calls:
                        tool_call_count += 1
                        log.debug(
                            f"Yielding tool call {tool_call_count}: {tool_call.function.name}"
                        )
                        yield ToolCall(
                            name=tool_call.function.name,
                            args=dict(tool_call.function.arguments),
                        )

                content = response.message.content or ""

                yield Chunk(
                    content=content,
                    done=response.done or False,
                )

                if response.done:
                    log.debug(
                        f"Streaming completed. Total chunks: {chunk_count}, tool calls: {tool_call_count}"
                    )
                    break

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Tool] = [],
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        """
        Generate a complete message from Ollama without streaming.

        Args:
            messages: The conversation history
            model: The model to use
            tools: Optional tools to make available to the model
            **kwargs: Additional parameters to pass to the Ollama API


        Returns:
            Message: The complete response message
        """
        log.debug(f"Generating complete message for model: {model}")
        log.debug(f"Non-streaming with {len(messages)} messages, {len(tools)} tools")
        self._log_api_request("chat", messages, model=model, tools=tools, **kwargs)

        async with get_ollama_client() as client:
            params = self._prepare_request_params(
                messages,
                model,
                tools,
                response_format=response_format,
                max_tokens=max_tokens,
            )
            params["stream"] = False
            log.debug("Making non-streaming chat request")

            response = await client.chat(**params)
            log.debug("Received complete response from Ollama")

            self._update_usage_stats(response)
            content = response.message.content or ""
            log.debug(f"Response content length: {len(content)}")

            tool_calls = None
            if response.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        name=tool_call.function.name,
                        args=dict(tool_call.function.arguments),
                    )
                    for tool_call in response.message.tool_calls
                ]
                log.debug(f"Response contains {len(tool_calls)} tool calls")
            else:
                log.debug("Response contains no tool calls")

            res = Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )

            self._log_api_response("chat", res)
            log.debug("Returning generated message")

            return res

    # Textual tools support removed

    def _process_image_content(self, image: ImageRef) -> str:
        """
        Process an image reference to a base64-encoded JPEG.
        Converts all images to JPEG format, resizes to 512x512 bounds,
        and returns as a base64 string without data URI prefix.

        DEPRECATED: Use nodetool.media.image.image_utils.image_ref_to_base64_jpeg instead.

        Args:
            image: The ImageRef object containing the image URI or data

        Returns:
            str: The processed image as a base64 string
        """
        import warnings

        log.debug("Processing image content (deprecated method)")
        warnings.warn(
            "_process_image_content is deprecated. Use nodetool.media.image.image_utils.image_ref_to_base64_jpeg instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = image_ref_to_base64_jpeg(image)
        log.debug(f"Processed image to base64 string of length: {len(result)}")
        return result

    def is_context_length_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        is_context_error = (
            "context length" in msg
            or "context window" in msg
            or "token limit" in msg
            or "request too large" in msg
            or "413" in msg
        )
        log.debug(f"Checking if error is context length error: {is_context_error}")
        return is_context_error

    def get_usage(self) -> dict:
        """Return the current accumulated token usage statistics."""
        log.debug(f"Getting usage stats: {self.usage}")
        return self.usage.copy()

    def reset_usage(self) -> None:
        """Reset the usage counters to zero."""
        log.debug("Resetting usage counters")
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


async def main():
    from nodetool.agents.tools.math_tools import CalculatorTool
    from nodetool.workflows.processing_context import ProcessingContext

    provider = OllamaProvider()
    tools = [CalculatorTool()]
    context = ProcessingContext()

    messages: list[Message] = [
        Message(
            role="system",
            content=(
                "You are a helpful assistant. Use tools when calculations are needed."
            ),
        ),
        Message(
            role="user",
            content=(
                "Please compute 12.3 * (7 - 5) + 10 / 2 and provide the numeric result."
            ),
        ),
    ]

    model_name = "gpt-oss:20b"

    response = await provider.generate_message(
        messages=messages,
        model=model_name,
        tools=tools,
    )
    print(response.content)

    while response.tool_calls:
        print(response.tool_calls)
        for tool_call in response.tool_calls:
            matching_tool = next((t for t in tools if t.name == tool_call.name), None)
            if matching_tool is None:
                continue
            tool_result = await matching_tool.process(context, tool_call.args or {})
            messages.append(
                Message(
                    role="tool",
                    name=matching_tool.name,
                    content=json.dumps(tool_result),
                )
            )

        response = await provider.generate_message(
            messages=messages,
            model=model_name,
            tools=tools,
        )
        print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
