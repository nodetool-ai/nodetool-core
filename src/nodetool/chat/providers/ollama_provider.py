"""
Ollama provider implementation for chat completions.

This module implements the ChatProvider interface for Ollama models,
handling message conversion, streaming, and tool integration.
"""

import json
import re
import ast
from typing import Any, AsyncGenerator, Sequence, Dict
from contextlib import asynccontextmanager

from ollama import AsyncClient, Client
from pydantic import BaseModel
import tiktoken

from nodetool.chat.providers.base import ChatProvider
from nodetool.agents.tools.base import Tool
from nodetool.common.environment import Environment
from nodetool.metadata.types import (
    Message,
    ToolCall,
    MessageImageContent,
    MessageTextContent,
    ImageRef,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


def get_ollama_sync_client() -> Client:
    """Get a sync client for the Ollama API."""
    api_url = Environment.get("OLLAMA_API_URL")
    assert api_url, "OLLAMA_API_URL not set"

    api_key = Environment.get("OLLAMA_API_KEY")
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}
    else:
        headers = {}
    return Client(api_url, headers=headers)


@asynccontextmanager
async def get_ollama_client() -> AsyncGenerator[AsyncClient, None]:
    """Get an AsyncClient for the Ollama API."""
    api_url = Environment.get("OLLAMA_API_URL")
    assert api_url, "OLLAMA_API_URL not set"

    api_key = Environment.get("OLLAMA_API_KEY")
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}
    else:
        headers = {}

    client = AsyncClient(api_url, headers=headers)
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

    def __init__(self, log_file=None):
        """Initialize the Ollama provider.

        Args:
            log_file (str, optional): Path to a file where API calls and responses will be logged.
                If None, no logging will be performed.
        """
        super().__init__()
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.log_file = log_file

    def get_max_token_limit(self, model: str) -> int:
        """Get the maximum token limit for a given model."""
        try:
            client = get_ollama_sync_client()

            # Construct the URL for the show endpoint
            model_info = client.show(model=model).modelinfo
            if model_info is None:
                return 4096

            if model_info.get("llama.context_length"):
                return model_info["llama.context_length"]
            else:
                # First try to get context length from model_info if available

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
        """
        Count the number of tokens in the message history.

        Args:
            messages: The messages to count tokens for

        Returns:
            int: The approximate token count
        """
        token_count = 0

        for msg in messages:
            # Count tokens in the message content
            if hasattr(msg, "content") and msg.content:
                if isinstance(msg.content, str):
                    token_count += len(self.encoding.encode(msg.content))
                elif isinstance(msg.content, list):
                    # For multi-modal content, just count the text parts
                    for part in msg.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            token_count += len(
                                self.encoding.encode(part.get("text", ""))
                            )

            # Count tokens in tool calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # Count function name
                    token_count += len(self.encoding.encode(tool_call.name))
                    # Count arguments
                    if isinstance(tool_call.args, dict):
                        token_count += len(
                            self.encoding.encode(json.dumps(tool_call.args))
                        )
                    else:
                        token_count += len(self.encoding.encode(str(tool_call.args)))

        return token_count

    def convert_message(
        self, message: Message, use_textual_tools: bool
    ) -> Dict[str, Any]:
        """
        Convert an internal message to Ollama's format.

        Args:
            message: The message to convert
        """
        if message.role == "tool":
            if use_textual_tools:
                # For textual tool calling, format tool responses as user messages with tool_output syntax
                if isinstance(message.content, BaseModel):
                    content = message.content.model_dump_json()
                else:
                    content = (
                        json.dumps(message.content)
                        if isinstance(message.content, (dict, list))
                        else str(message.content)
                    )

                return {"role": "user", "content": content}
            else:
                # Standard tool message format
                if isinstance(message.content, BaseModel):
                    content = message.content.model_dump_json()
                else:
                    content = (
                        json.dumps(message.content)
                        if isinstance(message.content, (dict, list))
                        else str(message.content)
                    )
                return {"role": "tool", "content": content, "name": message.name}
        elif message.role == "system":
            return {"role": "system", "content": message.content}
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
            return {
                "role": "assistant",
                "content": message.content or "",
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
        use_textual_tools: bool = True,
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
        # Use textual tool calling if configured
        if use_textual_tools and tools:
            # Modify the first system or user message to include tool instructions
            tool_instructions = self._create_textual_tools_prompt(tools)
            messages_list = list(messages)

            if isinstance(messages_list[-1].content, str):
                messages_list[-1].content += tool_instructions
            elif isinstance(messages_list[-1].content, list):
                messages_list[-1].content.append(
                    MessageTextContent(text=tool_instructions)
                )
            else:
                raise ValueError(
                    f"Unexpected message content type: {type(messages_list[-1].content)}"
                )

            # Use the modified messages and don't pass tools parameter
            ollama_messages = [
                self.convert_message(m, use_textual_tools) for m in messages_list
            ]

            params = {
                "model": model,
                "messages": ollama_messages,
                "options": {
                    "num_predict": max_tokens,
                    "num_ctx": context_window,
                },
            }

            # Don't include tools param for textual tool calling
            tools = []
        else:
            # Regular message conversion
            ollama_messages = [
                self.convert_message(m, use_textual_tools) for m in messages
            ]

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

        if model.startswith("granite") or model.startswith("qwen"):
            if "options" not in params:
                params["options"] = {}

            # Calculate appropriate context size based on token count
            min_ctx = 8192  # Keep current minimum
            max_ctx = 128000  # Maximum context of 128k
            suggested_ctx = self._count_tokens(messages)

            # Round up to nearest power of 2 for better performance
            # but cap at max_ctx and ensure at least min_ctx
            power = 13  # 2^13 = 8192 (our minimum)
            while (1 << power) < suggested_ctx and (1 << power) < max_ctx:
                power += 1

            ctx_size = min(max_ctx, max((1 << power), min_ctx))
            params["options"]["num_ctx"] = ctx_size

        if response_format:
            if (
                response_format.get("type") == "json_schema"
                and "json_schema" in response_format
            ):
                schema = response_format["json_schema"]
                if not "schema" in schema:
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
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
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

        use_textual_tools = (
            len(tools) > 0 and model.startswith("gemma") and not response_format
        )

        async with get_ollama_client() as client:
            if use_textual_tools and tools:
                params = self._prepare_request_params(
                    messages,
                    model,
                    tools,
                    max_tokens=max_tokens,
                    use_textual_tools=use_textual_tools,
                    context_window=context_window,
                )
                params["stream"] = True

                completion = await client.chat(**params)
                buffer = ""

                async for response in completion:
                    # Track usage metrics when we receive the final response
                    if response.done:
                        self._update_usage_stats(response)

                    new_content = response.message.content or ""
                    buffer += new_content

                    tool_calls = self._extract_tool_calls(buffer)

                    if tool_calls:
                        for tool_call in tool_calls:
                            yield tool_call

                    yield Chunk(content=new_content, done=response.done or False)
            else:
                # Standard tool calling handling
                params = self._prepare_request_params(
                    messages,
                    model,
                    tools,
                    use_textual_tools=False,
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

        use_textual_tools = (
            len(tools) > 0 and model.startswith("gemma") and not response_format
        )

        async with get_ollama_client() as client:
            if use_textual_tools and tools:
                params = self._prepare_request_params(
                    messages,
                    model,
                    tools,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    use_textual_tools=use_textual_tools,
                    context_window=context_window,
                )
                params["stream"] = False

                # Call API without streaming
                response = await client.chat(**params)

                # Update token usage
                self._update_usage_stats(response)

                content = response.message.content or ""

                # Check for tool calls in the response
                tool_calls = self._extract_tool_calls(content)

                if tool_calls:
                    res = Message(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                    )
                else:
                    res = Message(
                        role="assistant",
                        content=content,
                    )
            else:
                # Standard tool calling handling
                params = self._prepare_request_params(
                    messages,
                    model,
                    tools,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    use_textual_tools=use_textual_tools,
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
                    content=response.message.content or "",
                    tool_calls=tool_calls,
                )

            self._log_api_response("chat", res)

            return res

    def _create_textual_tools_prompt(self, tools: Sequence[Tool]) -> str:
        """Create a textual prompt with tool instructions and function signatures."""
        prompt = """
You have access to functions. If you decide to invoke any of the function(s),
you MUST put it in the format of
{"name": function name, "parameters": dictionary of argument name and its value}

You SHOULD NOT include any other text in the response if you call a function

Following functions are available:
"""
        tool_schemas = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in tools
        ]
        return prompt + json.dumps(tool_schemas, indent=2)

    def _extract_tool_calls(self, text: str) -> list[ToolCall]:
        """
        Extract all tool calls from the model response text using JSON parsing.
        Always searches for matching braces to find JSON objects in the text.

        Returns:
            list[ToolCall]: A list of tool calls extracted from the response.
        """
        tool_calls = []

        # Always search for JSON objects by looking for matching braces
        start_idx = text.find("{")
        while start_idx != -1:
            # Find potential JSON objects
            brace_count = 1
            end_idx = start_idx + 1

            while end_idx < len(text) and brace_count > 0:
                if text[end_idx] == "{":
                    brace_count += 1
                elif text[end_idx] == "}":
                    brace_count -= 1
                end_idx += 1

            if brace_count == 0:  # Found a complete JSON object
                potential_json = text[start_idx:end_idx]
                try:
                    data = json.loads(potential_json)
                    if (
                        isinstance(data, dict)
                        and "name" in data
                        and "parameters" in data
                    ):
                        tool_calls.append(
                            ToolCall(
                                name=data["name"],
                                args=data["parameters"],
                            )
                        )
                except json.JSONDecodeError as e:
                    pass
            # Look for the next potential JSON object
            start_idx = text.find("{", end_idx)

        return tool_calls

    # Keep the original method for backward compatibility but make it use the new implementation
    def _extract_tool_call(self, text: str) -> ToolCall | None:
        """
        Extract the first tool call from the model response text.

        Returns:
            ToolCall: The first tool call if found, otherwise None.
        """
        tool_calls = self._extract_tool_calls(text)
        return tool_calls[0] if tool_calls else None

    def _extract_ast_value(self, node: ast.AST) -> Any:
        """
        Recursively extract Python literal values from AST nodes.

        Handles constants, lists, dictionaries, and nested structures.

        Args:
            node: The AST node to extract value from

        Returns:
            The Python value represented by the AST node
        """
        if isinstance(node, (ast.Constant, ast.Str, ast.Num)):
            return node.value
        elif isinstance(node, ast.List):
            return [self._extract_ast_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._extract_ast_value(key): self._extract_ast_value(value)
                for key, value in zip(node.keys, node.values)
                if key is not None and value is not None
            }
        elif isinstance(node, ast.Tuple):
            return tuple(self._extract_ast_value(elt) for elt in node.elts)
        elif isinstance(node, ast.Set):
            return {self._extract_ast_value(elt) for elt in node.elts}
        elif isinstance(node, ast.Name):
            # Handle special constants like True, False, None
            if node.id == "True":
                return True
            elif node.id == "False":
                return False
            elif node.id == "None":
                return None
            else:
                raise ValueError(f"Cannot extract value from name: {node.id}")
        else:
            raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

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


async def run_smoke_test():
    """Run a smoke test for textual tool calling with all possible AST value types."""

    class ComplexDataTool(Tool):
        name = "process_complex_data"
        description = "Process data of various types to test AST extraction."
        input_schema = {
            "type": "object",
            "properties": {
                "list_data": {"type": "array"},
                "dict_data": {"type": "object"},
                "tuple_data": {"type": "array"},
                "set_data": {"type": "array"},
                "bool_values": {"type": "array"},
                "none_value": {"type": "null"},
                "nested_data": {"type": "object"},
            },
        }

    class SimpleDataTool(Tool):
        name = "process_simple_data"
        description = "Process simple data types."
        input_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "number": {"type": "number"},
                "flag": {"type": "boolean"},
            },
        }

    # Initialize the provider
    provider = OllamaProvider()

    # Create test messages with prompts that will trigger multiple tool calls
    messages = [
        Message(role="system", content="You are a helpful AI assistant."),
        Message(
            role="user",
            content="""Please help me test multiple tool calls in a single message.
            
            First, call process_complex_data with:
            - list_data=[1, 2, 3, "text"]
            - dict_data={"key": "value", "number": 42}
            - nested_data={"list": [1, 2, {"nested": True}]}
            
            Then, call process_simple_data with:
            - text="Hello, world!"
            - number=42
            - flag=True
            
            Make sure to make both function calls in your response.
            """,
        ),
    ]

    tools = [
        ComplexDataTool(),
        SimpleDataTool(),
    ]

    print("Running smoke test for multiple textual tool calls in a single message...")

    try:
        # Non-streaming test
        print("\n=== Testing non-streaming API with multiple tool calls ===")
        response = await provider.generate_message(
            messages=messages,
            model="gemma3:4b",
            tools=tools,
        )

        print(f"Response content: {response.content}")
        if response.tool_calls:
            print(f"Number of tool calls extracted: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls):
                print(f"\nTool call #{i+1}: {tool_call.name}")
                print(f"Args: {tool_call.args}")
                print(
                    f"Args types: {', '.join([f'{k}: {type(v).__name__}' for k, v in tool_call.args.items()])}"
                )

        # Streaming test
        print("\n=== Testing streaming API with multiple tool calls ===")
        chunks = []
        tool_calls_seen = []

        async for chunk in provider.generate_messages(
            messages=messages,
            model="gemma3:12b",
            tools=tools,
        ):
            if isinstance(chunk, Chunk):
                chunks.append(chunk.content)
                print(chunk.content, end="", flush=True)
            elif isinstance(chunk, ToolCall):
                tool_calls_seen.append(chunk)
                print(f"\nTool Call: {chunk.name}")
                print(f"Args: {chunk.args}")
                print(
                    f"Args types: {', '.join([f'{k}: {type(v).__name__}' for k, v in chunk.args.items()])}"
                )

        print("\n\nTest complete.")
        print(f"Saw {len(tool_calls_seen)} tool calls during streaming.")

        # Test for different tool names to verify multiple distinct calls were made
        tool_names = set(call.name for call in tool_calls_seen)
        print(f"Distinct tool names called: {', '.join(tool_names)}")
        if len(tool_names) > 1:
            print("SUCCESS: Multiple different tools were called!")

        # Check if we have both expected tool calls
        expected_tools = {"process_complex_data", "process_simple_data"}
        if expected_tools.issubset(tool_names):
            print("SUCCESS: All expected tools were called!")
        else:
            print(
                f"WARNING: Not all expected tools were called. Missing: {expected_tools - tool_names}"
            )

    except Exception as e:
        print(f"Error during smoke test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_smoke_test())
