"""
HuggingFace provider implementation for chat completions.

This module implements the ChatProvider interface for HuggingFace models using their
Inference Providers API with the AsyncInferenceClient from huggingface_hub.
"""

import json
import asyncio
import traceback
from typing import Any, AsyncGenerator, Literal, Sequence
import httpx

from huggingface_hub import AsyncInferenceClient

from nodetool.chat.providers.base import ChatProvider
from nodetool.agents.tools.base import Tool
from nodetool.config.logging_config import get_logger
import base64
from nodetool.media.image.image_utils import image_data_to_base64_jpeg
from nodetool.io.media_fetch import fetch_uri_bytes_and_mime_sync
from nodetool.metadata.types import (
    Message,
    Provider,
    ToolCall,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
)
from nodetool.config.environment import Environment
from nodetool.workflows.base_node import ApiKeyMissingError
from nodetool.workflows.types import Chunk
from pydantic import BaseModel

log = get_logger(__name__)

PROVIDER_T = Literal[
    "black-forest-labs",
    "cerebras",
    "cohere",
    "fal-ai",
    "featherless-ai",
    "fireworks-ai",
    "groq",
    "hf-inference",
    "hyperbolic",
    "nebius",
    "novita",
    "nscale",
    "openai",
    "replicate",
    "sambanova",
    "together",
]


class HuggingFaceProvider(ChatProvider):
    """
    HuggingFace implementation of the ChatProvider interface.

    Uses the HuggingFace Inference Providers API via AsyncInferenceClient from huggingface_hub.
    This provider works with various inference providers (Cerebras, Cohere, Fireworks, etc.)
    that support the OpenAI-compatible chat completion format.

    HuggingFace's message structure follows the OpenAI format:

    1. Message Format:
       - Each message is a dict with "role" and "content" fields
       - Role can be: "system", "user", "assistant", or "tool"
       - Content contains the message text (string) or content blocks (for multimodal)

    2. Tool Calls:
       - When a model wants to call a tool, the response includes a "tool_calls" field
       - Each tool call contains:
         - "id": A unique identifier for the tool call
         - "function": An object with "name" and "arguments" (JSON string)

    3. Response Structure:
       - response.choices[0].message contains the model's response
       - It includes fields like "role", "content", and optionally "tool_calls"
       - response.usage contains token usage statistics

    For more details, see: https://huggingface.co/docs/hugs/en/guides/function-calling#using-tools-function-definitions
    """

    provider: Provider = Provider.HuggingFace

    def __init__(self, inference_provider: PROVIDER_T | None = None):
        """Initialize the HuggingFace provider with AsyncInferenceClient."""
        super().__init__()
        env = Environment.get_environment()
        self.api_key = env.get("HF_TOKEN")
        self.inference_provider = inference_provider

        if not self.api_key:
            log.error("HF_TOKEN or HUGGINGFACE_API_KEY is not set")
            raise ApiKeyMissingError("HF_TOKEN or HUGGINGFACE_API_KEY is not set")

        # Initialize the AsyncInferenceClient
        if self.inference_provider:
            log.debug(
                f"Creating AsyncInferenceClient with provider: {self.inference_provider}"
            )
            self.client = AsyncInferenceClient(
                api_key=self.api_key,
                provider=self.inference_provider,
            )
        else:
            log.debug("Creating AsyncInferenceClient with default provider")
            self.client = AsyncInferenceClient(api_key=self.api_key)

        self.cost = 0.0
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        log.debug(
            f"HuggingFaceProvider initialized with provider: {inference_provider or 'default'}"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        log.debug("Entering async context manager")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - properly close client."""
        log.debug("Exiting async context manager")
        await self.close()

    async def close(self):
        """Close the async client properly."""
        log.debug("Closing async client")
        if hasattr(self.client, "close"):
            await self.client.close()
            log.debug("Async client closed successfully")
        else:
            log.debug("Client does not have close method")

    def get_container_env(self) -> dict[str, str]:
        env_vars = {}
        if self.api_key:
            env_vars["HF_TOKEN"] = self.api_key
        if hasattr(self, "inference_provider") and self.inference_provider:
            env_vars["HUGGINGFACE_PROVIDER"] = self.inference_provider
        log.debug(f"Container environment variables: {list(env_vars.keys())}")
        return env_vars

    def get_context_length(self, model: str) -> int:
        """Get the maximum token limit for a given model."""
        log.debug(f"Getting context length for model: {model}")

        # Common HuggingFace model limits - this can be expanded based on specific models
        if "llama" in model.lower():
            log.debug("Using context length: 32768 (Llama)")
            return 32768  # Many Llama models support 32k context
        elif "qwen" in model.lower():
            log.debug("Using context length: 32768 (Qwen)")
            return 32768  # Qwen models often support large context
        elif "phi" in model.lower():
            log.debug("Using context length: 128000 (Phi)")
            return 128000  # Phi-4 supports 128k context
        elif "smol" in model.lower():
            log.debug("Using context length: 8192 (SmolLM)")
            return 8192  # SmolLM models typically have smaller context
        elif "gemma" in model.lower():
            log.debug("Using context length: 8192 (Gemma)")
            return 8192  # Gemma models typically support 8k context
        elif "deepseek" in model.lower():
            log.debug("Using context length: 32768 (DeepSeek)")
            return 32768  # DeepSeek models often support large context
        elif "mistral" in model.lower():
            log.debug("Using context length: 32768 (Mistral)")
            return 32768  # Mistral models support 32k context
        else:
            log.debug("Using default context length: 8192")
            return 8192  # Conservative default

    def convert_message(self, message: Message) -> dict:
        """Convert an internal message to HuggingFace's OpenAI-compatible format."""
        log.debug(f"Converting message with role: {message.role}")

        if message.role == "tool":
            log.debug(f"Converting tool message, tool_call_id: {message.tool_call_id}")
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
            assert message.tool_call_id is not None, "Tool call ID must not be None"
            return {
                "role": "tool",
                "content": content,
                "tool_call_id": message.tool_call_id,
            }
        elif message.role == "system":
            log.debug("Converting system message")
            return {
                "role": "system",
                "content": str(message.content),
            }
        elif message.role == "user":
            log.debug("Converting user message")
            if isinstance(message.content, str):
                log.debug("User message has string content")
                return {"role": "user", "content": message.content}
            elif message.content is not None:
                # Handle multimodal content
                log.debug(f"Converting {len(message.content)} content parts")
                content = []
                for part in message.content:
                    if isinstance(part, MessageTextContent):
                        content.append({"type": "text", "text": part.text})
                    elif isinstance(part, MessageImageContent):
                        # Normalize image inputs to data URI using shared helpers
                        if part.image.uri:
                            uri = part.image.uri
                            if uri.startswith("http"):
                                # Keep remote URLs as-is for HF compatibility
                                log.debug(
                                    f"Adding image content with URL: {uri[:50]}..."
                                )
                                content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": uri},
                                    }
                                )
                            else:
                                # Convert non-http URIs to data URI
                                log.debug(
                                    f"Converting non-HTTP image via helper: {uri[:50]}..."
                                )
                                mime, data = fetch_uri_bytes_and_mime_sync(uri)
                                b64 = base64.b64encode(data).decode("utf-8")
                                content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime};base64,{b64}"
                                        },
                                    }
                                )
                        elif part.image.data:
                            # Convert raw bytes to JPEG base64 data URI
                            log.debug("Converting raw image data to JPEG base64")
                            b64 = image_data_to_base64_jpeg(part.image.data)
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}"
                                    },
                                }
                            )
                log.debug(f"Converted to {len(content)} content parts")
                return {"role": "user", "content": content}
            else:
                log.debug("User message has no content")
                return {"role": "user", "content": ""}
        elif message.role == "assistant":
            log.debug("Converting assistant message")
            result: dict[str, Any] = {"role": "assistant"}

            if message.content:
                result["content"] = str(message.content)
                log.debug("Assistant message has content")
            else:
                log.debug("Assistant message has no content")

            if message.tool_calls:
                log.debug(f"Assistant message has {len(message.tool_calls)} tool calls")
                result["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": (
                                json.dumps(tool_call.args)
                                if isinstance(tool_call.args, dict)
                                else str(tool_call.args)
                            ),
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            else:
                log.debug("Assistant message has no tool calls")

            return result
        else:
            log.error(f"Unsupported message role: {message.role}")
            raise ValueError(f"Unsupported message role: {message.role}")

    def format_tools(self, tools: Sequence[Tool]) -> list[dict]:
        """Format tools for HuggingFace API (OpenAI-compatible format)."""
        log.debug(f"Formatting {len(tools)} tools for HuggingFace API")
        formatted_tools = []
        for tool in tools:
            log.debug(f"Formatting tool: {tool.name}")
            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        log.debug(f"Formatted {len(formatted_tools)} tools")
        return formatted_tools

    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        context_window: int = 4096,
        response_format: dict | None = None,
        **kwargs,
    ) -> Message:
        """
        Generate a single message completion from HuggingFace using AsyncInferenceClient.

        Args:
            messages: Sequence of Message objects representing the conversation
            model: Model identifier (can be repo_id like "microsoft/Phi-4-mini-flash-reasoning")
            tools: Available tools for the model to use
            max_tokens: Maximum number of tokens to generate
            context_window: Maximum number of tokens to keep in context
            response_format: Format of the response
            **kwargs: Additional provider-specific parameters

        Returns:
            A message returned by the provider.
        """
        log.debug(f"Generating message for model: {model}")
        log.debug(
            f"Input: {len(messages)} messages, {len(tools)} tools, max_tokens: {max_tokens}"
        )

        # Convert messages to HuggingFace format
        log.debug("Converting messages to HuggingFace format")
        hf_messages = []
        for message in messages:
            converted = self.convert_message(message)
            if converted:  # Skip None messages
                hf_messages.append(converted)
        log.debug(f"Converted to {len(hf_messages)} HuggingFace messages")

        # Prepare request parameters - using HuggingFace's chat_completion method
        request_params: dict[str, Any] = {
            "messages": hf_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        log.debug(f"Request params: max_tokens={max_tokens}, stream=False")

        # Add tools if provided (following HuggingFace docs format)
        if tools:
            request_params["tools"] = self.format_tools(tools)
            request_params["tool_choice"] = "auto"  # As per HF docs
            log.debug("Added tools and tool_choice to request")

        # Add response format if specified
        if response_format:
            request_params["response_format"] = response_format
            log.debug("Added response format to request")

        max_retries = 3
        base_delay = 1.0
        log.debug(f"Starting API call with max_retries={max_retries}")

        completion: Any | None = None

        for attempt in range(max_retries + 1):
            try:
                log.debug(f"API call attempt {attempt + 1}/{max_retries + 1}")
                completion = await self.client.chat_completion(
                    model=model, **request_params
                )
                log.debug("API call successful")
                break
            except Exception as e:
                error_str = str(e).lower()
                log.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                # Do not retry on client-side errors (4xx), including 429, 413, 404.
                status = getattr(getattr(e, "response", None), "status_code", None)
                body_text = getattr(getattr(e, "response", None), "text", None)
                if isinstance(status, int) and 400 <= status < 500:
                    log.error(f"Non-retryable client error {status}; aborting retries")
                    raise Exception(f"{status} {body_text or str(e)}")
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    log.debug(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    log.error(f"All {max_retries + 1} attempts failed")
                    traceback.print_exc()
                    # Include status code/body text when available for clearer errors/tests
                    if status is not None:
                        raise Exception(f"{status} {body_text or str(e)}")
                    raise Exception(str(e))

        if completion is None:
            raise RuntimeError("HuggingFace chat completion did not return a response")

        # Update usage statistics if available
        if hasattr(completion, "usage") and completion.usage:
            log.debug("Processing usage statistics")
            self.usage["prompt_tokens"] = completion.usage.prompt_tokens or 0
            self.usage["completion_tokens"] = completion.usage.completion_tokens or 0
            self.usage["total_tokens"] = completion.usage.total_tokens or 0
            log.debug(f"Updated usage: {self.usage}")

        # Extract the response message
        choice = completion.choices[0]
        message_data = choice.message
        log.debug(f"Response content length: {len(message_data.content or '')}")

        # Create the response message
        response_message = Message(
            role="assistant",
            content=message_data.content or "",
        )

        # Handle tool calls if present
        if hasattr(message_data, "tool_calls") and message_data.tool_calls:
            log.debug(f"Processing {len(message_data.tool_calls)} tool calls")
            tool_calls = []
            for tool_call in message_data.tool_calls:
                function = tool_call.function
                try:
                    # Parse arguments - they might be JSON string or dict
                    args = function.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                        log.debug(f"Parsed JSON arguments for tool: {function.name}")
                    else:
                        log.debug(f"Using dict arguments for tool: {function.name}")
                except (json.JSONDecodeError, AttributeError) as e:
                    log.warning(
                        f"Failed to parse arguments for tool {function.name}: {e}"
                    )
                    args = {}

                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=function.name,
                        args=args,
                    )
                )
            response_message.tool_calls = tool_calls
            log.debug(f"Added {len(tool_calls)} tool calls to response")
        else:
            log.debug("Response contains no tool calls")

        log.debug("Returning generated message")
        return response_message

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
        Generate message completions from HuggingFace, yielding chunks or tool calls.

        Uses AsyncInferenceClient's streaming capability for real-time token generation.

        Args:
            messages: Sequence of Message objects representing the conversation
            model: Model identifier
            tools: Available tools for the model to use
            max_tokens: Maximum number of tokens to generate
            context_window: Maximum number of tokens to keep in context
            response_format: Format of the response
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunk objects with content and completion status or ToolCall objects
        """
        log.debug(f"Starting streaming generation for model: {model}")
        log.debug(f"Streaming with {len(messages)} messages, {len(tools)} tools")

        # Convert messages to HuggingFace format
        log.debug("Converting messages to HuggingFace format")
        hf_messages = []
        for message in messages:
            converted = self.convert_message(message)
            if converted:  # Skip None messages
                hf_messages.append(converted)
        log.debug(f"Converted to {len(hf_messages)} HuggingFace messages")

        # Prepare request parameters for streaming
        request_params: dict[str, Any] = {
            "messages": hf_messages,
            "max_tokens": max_tokens,
            "stream": True,  # Enable streaming
        }
        log.debug("Prepared streaming request parameters")

        # Add tools if provided
        if tools:
            request_params["tools"] = self.format_tools(tools)
            request_params["tool_choice"] = "auto"
            log.debug("Added tools to streaming request")

        # Add response format if specified
        if response_format:
            request_params["response_format"] = response_format
            log.debug("Added response format to streaming request")

        # Create streaming completion using chat_completion method
        log.debug("Starting streaming API call")
        stream = await self.client.chat_completion(model=model, **request_params)
        log.debug("Streaming response initialized")

        # Track tool calls during streaming
        accumulated_tool_calls = {}
        chunk_count = 0

        try:
            async for chunk in stream:
                chunk_count += 1

                if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                    log.debug("Updating usage stats from streaming chunk")
                    usage = chunk.usage  # type: ignore[attr-defined]
                    self.usage["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
                    self.usage["completion_tokens"] = (
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                    self.usage["total_tokens"] = getattr(usage, "total_tokens", 0) or 0
                    log.debug(f"Updated usage: {self.usage}")

                choices = getattr(chunk, "choices", None)
                if not choices:
                    log.debug("Chunk has no choices, skipping")
                    continue

                choice = choices[0]
                delta = getattr(choice, "delta", None)
                log.debug(f"Processing delta with finish_reason: {choice.finish_reason}")

                if delta and getattr(delta, "content", None):
                    yield Chunk(
                        content=delta.content,
                        done=choice.finish_reason == "stop",
                    )

                if delta and getattr(delta, "tool_calls", None):
                    log.debug(f"Processing {len(delta.tool_calls)} tool call deltas")
                    for tool_call_delta in delta.tool_calls:
                        index = getattr(tool_call_delta, "index", 0)
                        log.debug(f"Processing tool call delta at index {index}")

                        if index not in accumulated_tool_calls:
                            accumulated_tool_calls[index] = {
                                "id": tool_call_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                            log.debug(f"Created new tool call at index {index}")

                        if tool_call_delta.id:
                            accumulated_tool_calls[index]["id"] = tool_call_delta.id
                            log.debug(f"Set tool call ID: {tool_call_delta.id}")

                        function_delta = getattr(tool_call_delta, "function", None)
                        if function_delta:
                            if getattr(function_delta, "name", None):
                                accumulated_tool_calls[index]["name"] = (
                                    function_delta.name or ""
                                )
                                log.debug(
                                    f"Set tool call name: {function_delta.name}"
                                )
                            if getattr(function_delta, "arguments", None):
                                accumulated_tool_calls[index]["arguments"] += (
                                    function_delta.arguments or ""
                                )
                                log.debug(
                                    "Added arguments to tool call: %s chars",
                                    len(function_delta.arguments or ""),
                                )

                if (
                    choices
                    and choice.finish_reason == "tool_calls"
                    and accumulated_tool_calls
                ):
                    log.debug(
                        "Streaming complete with %d tool calls",
                        len(accumulated_tool_calls),
                    )
                    for tool_call_data in accumulated_tool_calls.values():
                        try:
                            args = json.loads(tool_call_data["arguments"])
                            log.debug(
                                "Parsed arguments for tool: %s",
                                tool_call_data["name"],
                            )
                        except json.JSONDecodeError as err:
                            log.warning(
                                "Failed to parse arguments for tool %s: %s",
                                tool_call_data["name"],
                                err,
                            )
                            args = {}

                        yield ToolCall(
                            id=tool_call_data["id"],
                            name=tool_call_data["name"],
                            args=args,
                        )
                    log.debug("Yielded all accumulated tool calls")

                if choices and choice.finish_reason == "stop":
                    log.debug("Finish reason is stop; emitting synthetic done chunk")
                    yield Chunk(content="", done=True)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            body_text = getattr(getattr(e, "response", None), "text", None)
            if status is not None:
                raise Exception(f"{status} {body_text or str(e)}")
            raise

    def get_usage(self) -> dict:
        """Get token usage statistics."""
        log.debug(f"Getting usage stats: {self.usage}")
        return self.usage

    def reset_usage(self) -> None:
        """Reset token usage statistics."""
        log.debug("Resetting usage counters")
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def is_context_length_error(self, error: Exception) -> bool:
        """Check if the error is due to context length exceeding limits."""
        error_str = str(error).lower()
        is_context_error = any(
            phrase in error_str
            for phrase in [
                "context length",
                "maximum context",
                "token limit",
                "too long",
                "context size",
                "request too large",
                "413",
            ]
        )
        log.debug(f"Checking if error is context length error: {is_context_error}")
        return is_context_error


async def main():
    """
    Test function for the HuggingFaceProvider.

    This function demonstrates how to use the HuggingFaceProvider for both
    non-streaming and streaming completions with various models.
    """
    import os

    try:
        # Use async context manager to properly handle client cleanup
        async with HuggingFaceProvider() as provider:
            print("🚀 Initializing HuggingFace Provider...")
            print(
                f"✅ Provider initialized with {'custom' if provider.inference_provider else 'default'} provider"
            )

            # Test models - use models that are more likely to support tool calling
            test_models = [
                # "HuggingFaceTB/SmolLM3-3B",
                "deepseek-ai/DeepSeek-V3-0324"
            ]

            # Create test messages
            messages = [
                Message(
                    role="system",
                    content="You are a helpful assistant. Be concise and friendly.",
                ),
                Message(
                    role="user",
                    content="What is the capital of France? Answer in one sentence.",
                ),
            ]

            for model in test_models:
                print(f"\n🔍 Testing model: {model}")
                print("-" * 50)

                try:
                    # Test 1: Non-streaming completion
                    print("📝 Testing non-streaming completion...")

                    response = await provider.generate_message(
                        messages=messages,
                        model=model,
                        max_tokens=100,
                        temperature=0.7,
                    )

                    print(f"✅ Response: {response.content}")
                    print(f"📊 Usage: {provider.get_usage()}")

                    # Reset usage for next test
                    provider.reset_usage()

                    # Test 2: Streaming completion
                    print("\n🌊 Testing streaming completion...")
                    print("📝 Streaming response: ", end="", flush=True)

                    chunks = []
                    async for chunk in provider.generate_messages(
                        messages=messages,
                        model=model,
                        max_tokens=100,
                        temperature=0.7,
                    ):
                        if isinstance(chunk, Chunk):
                            print(chunk.content, end="", flush=True)
                            chunks.append(chunk.content)
                        elif isinstance(chunk, ToolCall):
                            print(f"\n🔧 Tool call: {chunk.name}({chunk.args})")

                    print(f"\n✅ Streaming complete. Total chunks: {len(chunks)}")
                    print(f"📊 Usage: {provider.get_usage()}")

                    # Reset for next model
                    provider.reset_usage()

                except Exception as e:
                    print(f"❌ Error testing {model}: {str(e)}")

            # Test 3: Tool calling example (following HF docs exactly)
            print(f"\n🔧 Testing tool calling with {test_models[0]}...")
            print("-" * 50)

            try:
                # Create a simple test tool following HF docs format exactly
                class TestTool(Tool):
                    name = "get_weather"
                    description = "Get the current weather for a location"
                    input_schema = {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use",
                            },
                        },
                        "required": ["location", "format"],
                    }

                tool = TestTool()

                tool_messages = [
                    Message(
                        role="system",
                        content="Don't make assumptions about values. Ask for clarification if needed.",
                    ),
                    Message(
                        role="user",
                        content="What's the weather like in Tokyo, Japan in celsius?",
                    ),
                ]

                response = await provider.generate_message(
                    messages=tool_messages,
                    model=test_models[0],
                    tools=[tool],
                    max_tokens=150,
                )

                if response.tool_calls:
                    print(f"🔧 Tool calls detected: {len(response.tool_calls)}")
                    for tool_call in response.tool_calls:
                        print(f"   - {tool_call.name}: {tool_call.args}")
                else:
                    print(f"💬 Regular response: {response.content}")

                print(f"📊 Usage: {provider.get_usage()}")

            except Exception as e:
                print(f"⚠️  Tool calling test failed: {str(e)}")

            print("\n🎉 All tests completed!")

    except Exception as e:
        print(f"❌ Failed to initialize provider: {str(e)}")
        print("Make sure HF_TOKEN is set and you have huggingface_hub installed:")
        print("pip install huggingface_hub")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
