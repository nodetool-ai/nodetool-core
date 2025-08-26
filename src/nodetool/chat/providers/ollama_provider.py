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
from nodetool.common.environment import Environment
from nodetool.metadata.types import (
    Message,
    Provider,
    ToolCall,
    MessageImageContent,
    MessageTextContent,
    ImageRef,
)
from nodetool.workflows.types import Chunk


def get_ollama_sync_client() -> Client:
    """Get a sync client for the Ollama API."""
    api_url = Environment.get("OLLAMA_API_URL")
    assert api_url, "OLLAMA_API_URL not set"

    return Client(api_url)


@asynccontextmanager
async def get_ollama_client() -> AsyncGenerator[AsyncClient, None]:
    """Get an AsyncClient for the Ollama API."""
    api_url = Environment.get("OLLAMA_API_URL")
    assert api_url, "OLLAMA_API_URL not set"

    client = AsyncClient(api_url)
    try:
        yield client
    finally:
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

    def get_container_env(self) -> dict[str, str]:
        env_vars = {}
        if self.api_url:
            env_vars["OLLAMA_API_URL"] = self.api_url
        return env_vars

    def get_context_length(self, model: str) -> int:
        """Get the maximum token limit for a given model."""
        try:
            client = get_ollama_sync_client()

            # Construct the URL for the show endpoint
            model_info = client.show(model=model).modelinfo
            if model_info is None:
                return 4096

            for key, value in model_info.items():
                if ".context_length" in key:
                    return int(value)

            # Otherwise, try to extract from modelfile parameters
            if model_info["modelfile"]:
                modelfile = model_info["modelfile"]
                param_match = re.search(r"PARAMETER\s+num_ctx\s+(\d+)", modelfile)
                if param_match:
                    return int(param_match.group(1))

            # Default fallback if we can't determine the context length
            return 4096
        except Exception as e:
            print(f"Error determining model context length: {e}")
            # Fallback to a reasonable default
            return 4096

    def _count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate token count for a sequence of messages."""
        return count_messages_tokens(messages, encoding=self.encoding)

    def convert_message(self, message: Message) -> Dict[str, Any]:
        """
        Convert an internal message to Ollama's format.

        Args:
            message: The message to convert
        """
        if message.role == "tool":
            # Standard tool message format
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            else:
                if isinstance(message.content, dict):
                    content = json.dumps(message.content)
                elif isinstance(message.content, list):
                    content = json.dumps(
                        [part.model_dump() for part in message.content]
                    )
                else:
                    content = str(message.content)
            return {"role": "tool", "content": content, "name": message.name}
        elif message.role == "system":
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
            return {"role": "system", "content": content}
        elif message.role == "user":
            assert message.content is not None, "User message content must not be None"
            message_dict: Dict[str, Any] = {"role": "user"}

            if isinstance(message.content, str):
                message_dict["content"] = message.content
            else:
                # Handle text content
                text_parts = [
                    part.text
                    for part in message.content
                    if isinstance(part, MessageTextContent)
                ]
                message_dict["content"] = "\n".join(text_parts)

                # Handle image content
                image_parts = [
                    self._process_image_content(part.image)
                    for part in message.content
                    if isinstance(part, MessageImageContent)
                ]
                if image_parts:
                    message_dict["images"] = image_parts

            return message_dict
        elif message.role == "assistant":
            message_dict = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.args,
                        },
                    }
                    for tool_call in message.tool_calls or []
                ],
            }

            # Handle None content (common when only tool calls are present)
            if message.content is None:
                message_dict["content"] = ""
            elif isinstance(message.content, str):
                message_dict["content"] = message.content
            else:
                text_parts = [
                    part.text
                    for part in message.content
                    if isinstance(part, MessageTextContent)
                ]
                message_dict["content"] = "\n".join(text_parts)

            return message_dict
        else:
            raise ValueError(f"Unknown message role {message.role}")

    def format_tools(self, tools: Sequence[Any]) -> list:
        """Convert tools to Ollama's format."""
        return [tool.tool_param() for tool in tools]

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
        # Regular message conversion only
        ollama_messages = [self.convert_message(m) for m in messages]

        params = {
            "model": model,
            "messages": ollama_messages,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": context_window,
            },
        }

        if len(tools) > 0:
            params["tools"] = self.format_tools(tools)

        if response_format:
            if (
                response_format.get("type") == "json_schema"
                and "json_schema" in response_format
            ):
                schema = response_format["json_schema"]
                if "schema" not in schema:
                    raise ValueError(
                        "schema is required in json_schema response format"
                    )
                params["format"] = schema["schema"]

        return params

    def _update_usage_stats(self, response):
        """Update token usage statistics from response."""
        prompt_tokens = getattr(response, "prompt_eval_count", 0)
        completion_tokens = getattr(response, "eval_count", 0)

        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += prompt_tokens + completion_tokens

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
        self._log_api_request("chat_stream", messages, model, tools)

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

            completion = await client.chat(**params)
            async for response in completion:
                # Track usage metrics when we receive the final response
                if response.done:
                    self._update_usage_stats(response)

                if response.message.tool_calls is not None:
                    for tool_call in response.message.tool_calls:
                        yield ToolCall(
                            name=tool_call.function.name,
                            args=dict(tool_call.function.arguments),
                        )
                yield Chunk(
                    content=response.message.content or "",
                    done=response.done or False,
                )

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Tool] = [],
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
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
        self._log_api_request("chat", messages, model, tools)

        async with get_ollama_client() as client:
            params = self._prepare_request_params(
                messages,
                model,
                tools,
                response_format=response_format,
                max_tokens=max_tokens,
            )
            params["stream"] = False

            response = await client.chat(**params)

            self._update_usage_stats(response)
            content = response.message.content or ""

            tool_calls = None
            if response.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        name=tool_call.function.name,
                        args=dict(tool_call.function.arguments),
                    )
                    for tool_call in response.message.tool_calls
                ]

            res = Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )

            self._log_api_response("chat", res)

            return res

    # Textual tools support removed

    def _process_image_content(self, image: ImageRef) -> str:
        """
        Process an image reference to a base64-encoded JPEG.
        Converts all images to JPEG format, resizes to 512x512 bounds,
        and returns as a base64 string without data URI prefix.

        Args:
            image: The ImageRef object containing the image URI or data

        Returns:
            str: The processed image as a base64 string
        """
        import base64
        import requests
        from urllib.parse import urlparse
        import io
        from PIL import Image
        import os

        def process_image_data(image_data: bytes) -> str:
            """Convert image data to resized JPEG and return as base64 string."""
            try:
                # Open image with PIL
                with Image.open(io.BytesIO(image_data)) as img:
                    # Convert to RGB if needed (removes alpha channel)
                    if img.mode in ("RGBA", "LA") or (
                        img.mode == "P" and "transparency" in img.info
                    ):
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        background.paste(
                            img, mask=img.split()[3] if img.mode == "RGBA" else None
                        )
                        img = background
                    elif img.mode != "RGB":
                        img = img.convert("RGB")

                    # Resize if needed
                    if img.width > 512 or img.height > 512:
                        img.thumbnail((512, 512), Image.Resampling.LANCZOS)

                    # Save as JPEG
                    output = io.BytesIO()
                    img.save(output, format="JPEG", quality=85)

                    # Base64 encode without data URI prefix
                    base64_data = base64.b64encode(output.getvalue()).decode("utf-8")
                    return base64_data
            except Exception as e:
                print(f"Error processing image: {e}")
                raise

        # Case 1: Image has data bytes
        if hasattr(image, "data") and image.data:
            try:
                return process_image_data(image.data)
            except Exception as e:
                print(f"Failed to process image from data: {e}")

        # Case 2: Already a base64 data URI
        if image.uri and image.uri.startswith("data:"):
            try:
                # Extract the base64 data and decode
                header, encoded = image.uri.split(",", 1)
                image_data = base64.b64decode(encoded)
                # Re-process to standardize format and size
                return process_image_data(image_data)
            except Exception as e:
                print(f"Failed to process base64 image: {e}")
                # Return original if processing fails, but strip data URI prefix
                if image.uri.startswith("data:"):
                    return image.uri.split(",", 1)[1]
                return image.uri

        # Case 3: URL
        if image.uri:
            parsed = urlparse(image.uri)
            if parsed.scheme in ("http", "https"):
                try:
                    response = requests.get(image.uri, timeout=10)
                    response.raise_for_status()
                    return process_image_data(response.content)
                except Exception as e:
                    print(
                        f"Failed to download or process image from URL {image.uri}: {e}"
                    )

            # Case 4: Local file
            elif parsed.scheme == "file" or not parsed.scheme:
                try:
                    file_path = image.uri
                    if file_path.startswith("file://"):
                        file_path = file_path[7:]

                    with open(os.path.expanduser(file_path), "rb") as f:
                        return process_image_data(f.read())
                except Exception as e:
                    print(f"Failed to read or process local image {image.uri}: {e}")

        # If all processing attempts fail, return an empty string rather than a data URI
        return "" if not image.uri else image.uri

    def is_context_length_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        return (
            "context length" in msg or "context window" in msg or "token limit" in msg
        )


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
