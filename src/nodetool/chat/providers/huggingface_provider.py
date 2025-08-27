"""
HuggingFace provider implementation for chat completions.

This module implements the ChatProvider interface for HuggingFace models using their
Inference Providers API with the AsyncInferenceClient from huggingface_hub.
"""

import json
import asyncio
import traceback
from typing import Any, AsyncGenerator, Literal, Sequence

from huggingface_hub import AsyncInferenceClient

from nodetool.chat.providers.base import ChatProvider
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import (
    Message,
    Provider,
    ToolCall,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
)
from nodetool.common.environment import Environment
from nodetool.workflows.base_node import ApiKeyMissingError
from nodetool.workflows.types import Chunk
from pydantic import BaseModel

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
            raise ApiKeyMissingError("HF_TOKEN or HUGGINGFACE_API_KEY is not set")

        # Initialize the AsyncInferenceClient
        if self.inference_provider:
            self.client = AsyncInferenceClient(
                api_key=self.api_key,
                provider=self.inference_provider,
            )
        else:
            # Let AsyncInferenceClient use default provider
            self.client = AsyncInferenceClient(api_key=self.api_key)

        self.cost = 0.0
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - properly close client."""
        await self.close()

    async def close(self):
        """Close the async client properly."""
        if hasattr(self.client, "close"):
            await self.client.close()

    def get_container_env(self) -> dict[str, str]:
        env_vars = {}
        if self.api_key:
            env_vars["HF_TOKEN"] = self.api_key
        if hasattr(self, "inference_provider"):
            env_vars["HUGGINGFACE_PROVIDER"] = self.inference_provider
        return env_vars

    def get_context_length(self, model: str) -> int:
        """Get the maximum token limit for a given model."""
        # Common HuggingFace model limits - this can be expanded based on specific models
        if "llama" in model.lower():
            return 32768  # Many Llama models support 32k context
        elif "qwen" in model.lower():
            return 32768  # Qwen models often support large context
        elif "phi" in model.lower():
            return 128000  # Phi-4 supports 128k context
        elif "smol" in model.lower():
            return 8192  # SmolLM models typically have smaller context
        elif "gemma" in model.lower():
            return 8192  # Gemma models typically support 8k context
        elif "deepseek" in model.lower():
            return 32768  # DeepSeek models often support large context
        elif "mistral" in model.lower():
            return 32768  # Mistral models support 32k context
        else:
            return 8192  # Conservative default

    def convert_message(self, message: Message) -> dict:
        """Convert an internal message to HuggingFace's OpenAI-compatible format."""
        if message.role == "tool":
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
            assert message.tool_call_id is not None, "Tool call ID must not be None"
            return {
                "role": "tool",
                "content": content,
                "tool_call_id": message.tool_call_id,
            }
        elif message.role == "system":
            return {
                "role": "system",
                "content": str(message.content),
            }
        elif message.role == "user":
            if isinstance(message.content, str):
                return {"role": "user", "content": message.content}
            elif message.content is not None:
                # Handle multimodal content
                content = []
                for part in message.content:
                    if isinstance(part, MessageTextContent):
                        content.append({"type": "text", "text": part.text})
                    elif isinstance(part, MessageImageContent):
                        # For image content, use image_url format
                        content.append(
                            {"type": "image_url", "image_url": {"url": part.image.uri}}
                        )
                return {"role": "user", "content": content}
            else:
                return {"role": "user", "content": ""}
        elif message.role == "assistant":
            result: dict[str, Any] = {"role": "assistant"}

            if message.content:
                result["content"] = str(message.content)

            if message.tool_calls:
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

            return result
        else:
            raise ValueError(f"Unsupported message role: {message.role}")

    def format_tools(self, tools: Sequence[Tool]) -> list[dict]:
        """Format tools for HuggingFace API (OpenAI-compatible format)."""
        formatted_tools = []
        for tool in tools:
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
        # Convert messages to HuggingFace format
        hf_messages = []
        for message in messages:
            converted = self.convert_message(message)
            if converted:  # Skip None messages
                hf_messages.append(converted)

        # Prepare request parameters - using HuggingFace's chat_completion method
        request_params: dict[str, Any] = {
            "messages": hf_messages,
            "max_tokens": max_tokens,
            "stream": False,
        }

        # Add tools if provided (following HuggingFace docs format)
        if tools:
            request_params["tools"] = self.format_tools(tools)
            request_params["tool_choice"] = "auto"  # As per HF docs

        # Add response format if specified
        if response_format:
            request_params["response_format"] = response_format

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                completion = await self.client.chat_completion(
                    model=model, **request_params
                )
                break
            except Exception as e:
                error_str = str(e).lower()
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                    continue
                else:
                    traceback.print_exc()
                    raise Exception(
                        f"HuggingFace Inference API request failed: {str(e)}"
                    )

        # Update usage statistics if available
        if hasattr(completion, "usage") and completion.usage:
            self.usage["prompt_tokens"] = completion.usage.prompt_tokens or 0
            self.usage["completion_tokens"] = completion.usage.completion_tokens or 0
            self.usage["total_tokens"] = completion.usage.total_tokens or 0

        # Extract the response message
        choice = completion.choices[0]
        message_data = choice.message

        # Create the response message
        response_message = Message(
            role="assistant",
            content=message_data.content or "",
        )

        # Handle tool calls if present
        if hasattr(message_data, "tool_calls") and message_data.tool_calls:
            tool_calls = []
            for tool_call in message_data.tool_calls:
                function = tool_call.function
                try:
                    # Parse arguments - they might be JSON string or dict
                    args = function.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                except (json.JSONDecodeError, AttributeError):
                    args = {}

                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=function.name,
                        args=args,
                    )
                )
            response_message.tool_calls = tool_calls

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
        # Convert messages to HuggingFace format
        hf_messages = []
        for message in messages:
            converted = self.convert_message(message)
            if converted:  # Skip None messages
                hf_messages.append(converted)

        # Prepare request parameters for streaming
        request_params: dict[str, Any] = {
            "messages": hf_messages,
            "max_tokens": max_tokens,
            "stream": True,  # Enable streaming
        }

        # Add tools if provided
        if tools:
            request_params["tools"] = self.format_tools(tools)
            request_params["tool_choice"] = "auto"

        # Add response format if specified
        if response_format:
            request_params["response_format"] = response_format

        # Create streaming completion using chat_completion method
        stream = await self.client.chat_completion(model=model, **request_params)

        # Track tool calls during streaming
        accumulated_tool_calls = {}

        async for chunk in stream:
            # Update usage statistics if available
            if hasattr(chunk, "usage") and chunk.usage:
                self.usage["prompt_tokens"] = chunk.usage.prompt_tokens or 0
                self.usage["completion_tokens"] = chunk.usage.completion_tokens or 0
                self.usage["total_tokens"] = chunk.usage.total_tokens or 0

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Handle content chunks
            if hasattr(delta, "content") and delta.content:
                yield Chunk(
                    content=delta.content,
                    done=choice.finish_reason == "stop",
                )

            # Handle tool call deltas
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    index = tool_call_delta.index

                    if index not in accumulated_tool_calls:
                        accumulated_tool_calls[index] = {
                            "id": tool_call_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }

                    # Accumulate tool call data
                    if tool_call_delta.id:
                        accumulated_tool_calls[index]["id"] = tool_call_delta.id

                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            accumulated_tool_calls[index][
                                "name"
                            ] = tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            accumulated_tool_calls[index][
                                "arguments"
                            ] += tool_call_delta.function.arguments

            # If streaming is complete and we have tool calls, yield them
            if choice.finish_reason == "tool_calls" and accumulated_tool_calls:
                for tool_call_data in accumulated_tool_calls.values():
                    try:
                        args = json.loads(tool_call_data["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    yield ToolCall(
                        id=tool_call_data["id"],
                        name=tool_call_data["name"],
                        args=args,
                    )

    def get_usage(self) -> dict:
        """Get token usage statistics."""
        return self.usage

    def reset_usage(self) -> None:
        """Reset token usage statistics."""
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def is_context_length_error(self, error: Exception) -> bool:
        """Check if the error is due to context length exceeding limits."""
        error_str = str(error).lower()
        return any(
            phrase in error_str
            for phrase in [
                "context length",
                "maximum context",
                "token limit",
                "too long",
                "context size",
            ]
        )


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
