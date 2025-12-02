"""
Ollama provider implementation for chat completions.

This module implements the ChatProvider interface for Ollama models,
handling message conversion, streaming, and tool integration.
"""

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Sequence

import tiktoken
from ollama import AsyncClient, Client
from pydantic import BaseModel

from nodetool.agents.tools.base import Tool
from nodetool.chat.token_counter import count_messages_tokens
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.media.image.image_utils import (
    image_ref_to_base64_jpeg,
)
from nodetool.metadata.types import (
    ImageRef,
    LanguageModel,
    Message,
    MessageImageContent,
    MessageTextContent,
    Provider,
    ToolCall,
)
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.openai_compat import OpenAICompat
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

log = get_logger(__name__)
log.setLevel(logging.DEBUG)

# Only register the provider if OLLAMA_API_URL is explicitly set
_ollama_api_url = Environment.get("OLLAMA_API_URL")


def _resolve_ollama_api_url(explicit: str | None = None) -> str:
    api_url = explicit or os.environ.get("OLLAMA_API_URL") or Environment.get("OLLAMA_API_URL")
    assert api_url, "OLLAMA_API_URL not set"
    return api_url


def get_ollama_sync_client(api_url: str | None = None) -> Client:
    """Get a sync client for the Ollama API."""
    resolved_url = _resolve_ollama_api_url(api_url)
    log.debug(f"Creating sync Ollama client with URL: {resolved_url}")

    return Client(resolved_url)


@asynccontextmanager
async def get_ollama_client(api_url: str | None = None) -> AsyncGenerator[AsyncClient, None]:
    """Get an AsyncClient for the Ollama API."""
    resolved_url = _resolve_ollama_api_url(api_url)
    log.debug(f"Creating async Ollama client with URL: {resolved_url}")

    client = AsyncClient(resolved_url)
    try:
        yield client
    finally:
        log.debug("Closing async Ollama client")
        await client._client.aclose()


class OllamaProvider(BaseProvider, OpenAICompat):
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

    provider_name: str = "ollama"

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["OLLAMA_API_URL"]

    def __init__(self, secrets: dict[str, str], log_file=None):
        """Initialize the Ollama provider.

        Args:
            log_file (str, optional): Path to a file where API calls and responses will be logged.
            If None, no logging will be performed.
        """
        super().__init__(secrets)
        api_url = secrets.get("OLLAMA_API_URL") if secrets else None
        self.api_url = api_url or Environment.get("OLLAMA_API_URL")
        if self.api_url:
            os.environ.setdefault("OLLAMA_API_URL", self.api_url)
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.log_file = log_file
        self._model_info_cache: Dict[str, Any] = {}
        log.debug(
            f"OllamaProvider initialized. API URL present: {bool(self.api_url)}, log_file: {log_file}"
        )

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        env_vars = {}
        if self.api_url:
            env_vars["OLLAMA_API_URL"] = self.api_url
        log.debug(f"Container environment variables: {list(env_vars.keys())}")
        return env_vars

    def _get_model_info(self, model: str) -> Any:
        """Get model info from cache or fetch it from Ollama API.

        Args:
            model: Model identifier string.

        Returns:
            Model info object from Ollama API.

        Raises:
            Exception: If model info cannot be fetched.
        """
        # Check cache first
        if model in self._model_info_cache:
            log.debug(f"Using cached model info for: {model}")
            return self._model_info_cache[model]

        # Fetch and cache
        log.debug(f"Fetching model info for: {model}")
        client = get_ollama_sync_client(self.api_url)
        model_info = client.show(model=model)
        self._model_info_cache[model] = model_info
        log.debug(f"Cached model info for: {model}")
        log.debug(f"Model info: {model_info}")
        return model_info

    def get_context_length(self, model: str) -> int:
        """Get the maximum token limit for a given model."""
        log.debug(f"Getting context length for model: {model}")
        try:
            # Use cached model info
            model_response = self._get_model_info(model)
            model_info = model_response.modelinfo
            log.debug(f"Model info: {model_info}")
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
            log.warning("Using default context length: 4096")
            return 4096
        except Exception as e:
            log.error(f"Error determining model context length: {e}")
            print(f"Error determining model context length: {e}")
            # Fallback to a reasonable default
            return 4096

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        Checks the model's capabilities to determine if it supports tools.

        Args:
            model: Model identifier string.

        Returns:
            True if the model has "tools" in its capabilities, False otherwise.
            Falls back to True if capabilities cannot be determined.
        """
        log.debug(f"Checking tool support for model: {model}")
        try:
            # Use cached model info
            model_info = self._get_model_info(model)

            # Check if capabilities field exists and contains "tools"
            if hasattr(model_info, "capabilities") and model_info.capabilities:
                has_tools = "tools" in model_info.capabilities
                log.debug(
                    f"Model {model} capabilities: {model_info.capabilities}, tools support: {has_tools}"
                )
                return has_tools

            # If capabilities field doesn't exist, assume tools are supported for backward compatibility
            log.debug(f"Model {model} has no capabilities field, assuming tool support")
            return True

        except Exception as e:
            log.warning(f"Error checking tool support for model {model}: {e}")
            # Default to True for backward compatibility
            log.debug(f"Defaulting to True for model {model} due to error")
            return True

    async def get_available_language_models(self) -> List[LanguageModel]:
        """
        Get available Ollama models.

        Returns models available in the local Ollama installation.
        Returns an empty list if Ollama is not available.

        Returns:
            List of LanguageModel instances for Ollama
        """
        try:
            async with get_ollama_client(self.api_url) as client:
                models_response = await client.list()
                models: List[LanguageModel] = []
                # The Ollama client returns an object with a .models attribute
                for model in models_response.models:
                    model_name = model.model
                    if model_name:
                        models.append(
                            LanguageModel(
                                id=model_name,
                                name=model_name,
                                provider=Provider.Ollama,
                            )
                        )
                log.debug(f"Fetched {len(models)} Ollama models")
                return models
        except Exception as e:
            log.error(f"Error fetching Ollama models: {e}")
            return []

    def _count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate token count for a sequence of messages."""
        token_count = count_messages_tokens(messages, encoding=self.encoding)
        log.debug(f"Estimated token count for {len(messages)} messages: {token_count}")
        return token_count

    def convert_message(
        self, message: Message, use_tool_emulation: bool = False
    ) -> Dict[str, Any]:
        """
        Convert an internal message to Ollama's format.

        Args:
            message: The message to convert
            use_tool_emulation: Whether to convert tool messages for emulation
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

            # For tool emulation, convert to user message with clear explanation
            if use_tool_emulation:
                emulated_content = (
                    f"The function {message.name}() returned the following result:\n{content}\n\n"
                    f"Use this result to answer the user's question. Do NOT call the function again."
                )
                log.debug("Converting tool message to user message for emulation")
                return {"role": "user", "content": emulated_content}
            else:
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
        context_window: int | None = None,
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

        if context_window is None:
            context_window = self.get_context_length(model)
        # Check if model supports native tool calling
        use_tool_emulation = False
        if len(tools) > 0 and not self.has_tool_support(model):
            log.info(
                f"Model {model} does not support native tool calling, using emulation"
            )
            use_tool_emulation = True

        # Prepare messages
        ollama_messages = []
        if use_tool_emulation and len(tools) > 0:
            # Add tool definitions to system message
            tool_definitions = self._format_tools_as_python(tools)
            tool_instruction = (
                "\n\n=== AVAILABLE FUNCTIONS ===\n"
                "You can call these functions by writing a function call on a single line.\n"
                "DO NOT write function definitions - only write function CALLS.\n\n"
                f"{tool_definitions}\n\n"
                "=== INSTRUCTIONS ===\n"
                "When you need to use a function:\n"
                "1. Write ONLY the function call, nothing else\n"
                "2. Use this exact format: function_name(param='value')\n"
                "3. Do NOT write 'def', 'return', or any other Python keywords\n"
                "4. After calling a function, wait for the result\n"
                "5. Once you receive a function result, use it in your final answer\n"
                "6. Do NOT call the same function twice\n\n"
                "Example conversation:\n"
                "User: What is 5 + 3?\n"
                "You: calculator(expression='5 + 3')\n"
                "[System returns: {'result': 8}]\n"
                "You: The answer is 8."
            )

            # Find or create system message
            modified_messages = list(messages)
            has_system = any(m.role == "system" for m in modified_messages)

            if has_system:
                # Append to existing system message
                for i, msg in enumerate(modified_messages):
                    if msg.role == "system":
                        existing_content = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        modified_messages[i] = Message(
                            role="system", content=existing_content + tool_instruction
                        )
                        break
            else:
                # Prepend new system message
                modified_messages = [
                    Message(role="system", content=tool_instruction.strip()),
                    *modified_messages,
                ]

            ollama_messages = [
                self.convert_message(m, use_tool_emulation=True)
                for m in modified_messages
            ]
            log.debug("Using tool emulation: added tool definitions to system message")
        else:
            # Regular message conversion
            ollama_messages = [
                self.convert_message(m, use_tool_emulation=False) for m in messages
            ]

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

        # Add tools only if native support is available
        if len(tools) > 0 and not use_tool_emulation:
            params["tools"] = self.format_tools(tools)
            log.debug(f"Added {len(tools)} tools to request (native support)")

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

        # Determine if we're using tool emulation
        use_tool_emulation = len(tools) > 0 and not self.has_tool_support(model)
        if use_tool_emulation:
            log.info(f"Using tool emulation for model {model}")

        async with get_ollama_client(self.api_url) as client:
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
            accumulated_content = ""  # For tool emulation parsing

            async for response in completion:
                chunk_count += 1

                # Track usage metrics when we receive the final response
                if response.done:
                    log.debug("Final chunk received, updating usage stats")
                    self._update_usage_stats(response)

                # Handle native tool calls
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

                # Accumulate content for emulation parsing
                if use_tool_emulation:
                    accumulated_content += content

                yield Chunk(
                    content=content,
                    done=response.done or False,
                )

                if response.done:
                    # Parse emulated tool calls from accumulated content
                    if use_tool_emulation and accumulated_content:
                        log.debug("Parsing emulated tool calls from response")
                        emulated_calls = self._parse_function_calls(
                            accumulated_content, tools
                        )
                        for tool_call in emulated_calls:
                            tool_call_count += 1
                            log.debug(f"Yielding emulated tool call: {tool_call.name}")
                            yield tool_call

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

        # Determine if we're using tool emulation
        use_tool_emulation = len(tools) > 0 and not self.has_tool_support(model)
        if use_tool_emulation:
            log.info(f"Using tool emulation for model {model}")

        async with get_ollama_client(self.api_url) as client:
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

            # Handle native tool calls
            if response.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        name=tool_call.function.name,
                        args=dict(tool_call.function.arguments),
                    )
                    for tool_call in response.message.tool_calls
                ]
                log.debug(f"Response contains {len(tool_calls)} native tool calls")
            # Handle emulated tool calls
            elif use_tool_emulation and content:
                log.debug("Parsing emulated tool calls from response")
                emulated_calls = self._parse_function_calls(content, tools)
                if emulated_calls:
                    tool_calls = emulated_calls
                    log.debug(
                        f"Response contains {len(tool_calls)} emulated tool calls"
                    )
                else:
                    log.debug("Response contains no tool calls")
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

    provider = OllamaProvider(secrets={})
    tools = [CalculatorTool()]
    context = ProcessingContext()

    # Using gemma3:4b which doesn't support native tool calling
    # This will automatically use tool calling emulation
    model_name = "gemma3:4b"

    # Check if model supports native tool calling
    has_native_tools = provider.has_tool_support(model_name)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Native tool support: {has_native_tools}")
    print(f"Using tool emulation: {not has_native_tools}")
    print(f"{'='*60}\n")

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

    response = await provider.generate_message(
        messages=messages,
        model=model_name,
        tools=tools,
    )
    print("\n--- Initial Response ---")
    print(f"Content: {response.content}")
    print(f"Tool calls: {response.tool_calls}\n")

    iteration = 1
    max_iterations = 10
    while response.tool_calls and iteration <= max_iterations:
        print(f"\n--- Tool Execution (Iteration {iteration}/{max_iterations}) ---")
        for tool_call in response.tool_calls:
            print(f"Tool: {tool_call.name}")
            print(f"Args: {tool_call.args}")

            matching_tool = next((t for t in tools if t.name == tool_call.name), None)
            if matching_tool is None:
                print(f"Warning: Tool {tool_call.name} not found")
                continue

            tool_result = await matching_tool.process(context, tool_call.args or {})
            print(f"Result: {tool_result}")

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
        print("\n--- Response After Tool Call ---")
        print(f"Content: {response.content}")
        print(f"Tool calls: {response.tool_calls}")
        iteration += 1

    if iteration > max_iterations:
        print(f"\n⚠️  WARNING: Reached max iterations ({max_iterations}). Stopping.\n")

    print(f"\n{'='*60}")
    print(f"Final Answer: {response.content}")
    print(f"{'='*60}\n")

    # Test 2: Response format (structured output)
    print(f"\n{'='*60}")
    print("TEST 2: Response Format (Structured Output)")
    print(f"{'='*60}\n")

    # Define a JSON schema for structured output
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "calculation_result",
            "schema": {
                "type": "object",
                "properties": {
                    "calculation": {"type": "string"},
                    "result": {"type": "number"},
                    "explanation": {"type": "string"},
                },
                "required": ["calculation", "result", "explanation"],
            },
        },
    }

    messages_json = [
        Message(
            role="system",
            content="You are a helpful math assistant. Always respond with valid JSON.",
        ),
        Message(
            role="user",
            content="Calculate the area of a circle with radius 5. Explain your work.",
        ),
    ]

    print(f"Using response_format: {response_format['json_schema']['name']}")
    print(f"Schema: {json.dumps(response_format['json_schema']['schema'], indent=2)}\n")

    response_json = await provider.generate_message(
        messages=messages_json,
        model=model_name,
        response_format=response_format,
    )

    print("--- Structured Response ---")
    print(f"Content:\n{response_json.content}\n")

    # Try to parse the JSON
    try:
        content_str = (
            response_json.content
            if isinstance(response_json.content, str)
            else str(response_json.content)
        )
        parsed = json.loads(content_str)
        print("✅ Valid JSON!")
        print(f"Calculation: {parsed.get('calculation')}")
        print(f"Result: {parsed.get('result')}")
        print(f"Explanation: {parsed.get('explanation')}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")

    print(f"\n{'='*60}\n")


# Conditionally register the provider only if OLLAMA_API_URL is set
if _ollama_api_url:
    register_provider(Provider.Ollama)(OllamaProvider)
else:
    log.debug("Ollama provider not registered: OLLAMA_API_URL not set")


if __name__ == "__main__":
    asyncio.run(main())
