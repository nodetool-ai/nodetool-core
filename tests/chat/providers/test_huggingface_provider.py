"""
Tests for HuggingFace provider with comprehensive API response mocking.

This module tests the HuggingFace provider implementation including:
- Text Generation Inference (TGI) server integration
- Hugging Face Hub model access
- Inference API and Endpoints
- Custom model deployment
- OpenAI-compatible messaging

HuggingFace Text Generation Inference API Documentation (2024):
URLs:
- https://huggingface.co/docs/text-generation-inference/messages_api
- https://huggingface.github.io/text-generation-inference/
- https://github.com/huggingface/text-generation-inference

HuggingFace Text Generation Inference (TGI) provides high-performance inference for LLMs.

Text Generation Inference (TGI) Features:
- Production-ready serving for popular open-source LLMs
- OpenAI-compatible Messages API (v1.4.0+)
- Optimized inference with Flash Attention and Paged Attention
- Support for Llama, Falcon, StarCoder, BLOOM, GPT-NeoX, and more
- Used in production for Hugging Chat and Inference API

OpenAI-Compatible Messages API:
- Endpoint: /v1/chat/completions
- Full compatibility with OpenAI Chat Completion API
- Smooth migration path from OpenAI to open-source models
- Standard request/response format

Key Request Parameters:
- model: Model identifier (often "tgi" for TGI servers)
- messages: Array of message objects with role and content
- stream: Boolean for streaming responses
- max_tokens: Maximum response length
- temperature: Sampling temperature (0.0-2.0)
- top_p: Nucleus sampling (0.0-1.0)
- top_k: Top-k sampling
- repetition_penalty: Repetition penalty factor
- stop: Stop sequences

Response Format:
- Same as OpenAI Chat Completions API
- id: Completion identifier
- object: "chat.completion" or "chat.completion.chunk"
- model: Model name
- choices: Array with message content and finish_reason
- usage: Token statistics (prompt_tokens, completion_tokens, total_tokens)

TGI Performance Features:
- Flash Attention for memory efficiency
- Paged Attention for dynamic batching
- Logits warper (temperature, top-p, top-k, repetition penalty)
- Guidance and JSON structured output
- Continuous batching for high throughput
- Tensor parallelism for large models

Supported Models (Popular):
- Meta Llama 2 and 3 families
- Mistral and Mixtral models
- CodeLlama for code generation
- Falcon models
- StarCoder for code completion
- BLOOM multilingual models
- Phi models for efficiency
- Gemma models

Deployment Options:
- Docker containers for easy deployment
- Kubernetes and cloud-native scaling
- GPU acceleration (CUDA, ROCm)
- CPU inference for smaller models
- Quantization support (GPTQ, AWQ, GGUF)

HuggingFace Hub Integration:
- Direct model loading from Hub
- Automatic model downloading
- Token-gated model access
- Custom model deployment
- Model versioning and updates

Client Libraries:
- huggingface_hub Python library
- InferenceClient for easy API access
- Integration with transformers library
- LangChain and LlamaIndex compatibility

Inference Endpoints:
- Managed inference infrastructure
- Auto-scaling based on demand
- Regional deployment options
- Custom environment configuration
- Monitoring and logging

Advanced Features:
- Custom stopping criteria
- Logits processors and warpers
- Custom sampling strategies
- Multi-turn conversation support
- Context window management

Performance Optimizations:
- Continuous batching for throughput
- Speculative decoding for latency
- KV cache management
- Memory pooling and optimization
- Dynamic sequence length batching

Error Handling:
- 400: Invalid request format
- 401: Authentication required (for private models)
- 404: Model not found
- 413: Request too large (context limit exceeded)
- 429: Rate limit exceeded
- 500: Server error
- 503: Service unavailable (model loading)

Integration Patterns:
- REST API calls to TGI server
- Python client with huggingface_hub
- OpenAI SDK with custom base_url
- Streaming responses for real-time UX
- Batch processing for efficiency

Model Management:
- Automatic model caching
- Warm-up and preloading
- Multi-model serving
- A/B testing capabilities
- Resource allocation and limits
"""

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import httpx
import pytest
from huggingface_hub import AsyncInferenceClient

from nodetool.metadata.types import Message, MessageTextContent
from nodetool.providers.huggingface_provider import HuggingFaceProvider

from .test_base_provider import BaseProviderTest, ResponseFixtures


class TestHuggingFaceProvider(BaseProviderTest):
    """Test suite for HuggingFace provider with realistic API response mocking."""

    @property
    def provider_class(self):
        return HuggingFaceProvider

    @property
    def provider_name(self):
        return "huggingface"

    def _get_custom_test_secrets(self) -> Dict[str, str]:
        """Override to provide HuggingFace-specific test secrets."""
        return {
            "HF_TOKEN": "hf_test_token_12345",
        }

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Override to add HuggingFace-specific kwargs."""
        # Get the base kwargs from registration
        kwargs = super()._get_provider_kwargs()
        # Ensure we have the inference_provider for tests
        # (use hf-inference as default for unit tests, as it doesn't need a running server)
        if "inference_provider" not in kwargs:
            kwargs["inference_provider"] = "hf-inference"
        return kwargs

    def create_huggingface_response(self, content: str = "Hello, world!") -> Dict[str, Any]:
        """Create a realistic HuggingFace TGI API response."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }

    def create_huggingface_streaming_responses(self, text: str = "Hello world!") -> List[str]:
        """Create realistic HuggingFace TGI streaming response chunks."""
        chunks = []
        words = text.split()

        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            content = word + (" " if not is_last else "")

            chunk = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": "stop" if is_last else None,
                    }
                ],
            }

            if is_last:
                chunk["usage"] = {
                    "prompt_tokens": 10,
                    "completion_tokens": len(words),
                    "total_tokens": 10 + len(words),
                }

            chunks.append(f"data: {json.dumps(chunk)}")

        chunks.append("data: [DONE]")
        return chunks

    def create_huggingface_error(self, error_type: str = "model_not_found"):
        """Create realistic HuggingFace API errors."""
        if error_type == "model_not_found":
            return httpx.HTTPStatusError(
                message="Model not found",
                request=MagicMock(),
                response=MagicMock(status_code=404, text="Model not found"),
            )
        elif error_type == "context_length" or error_type == "token_limit":
            return httpx.HTTPStatusError(
                message="Request too large",
                request=MagicMock(),
                response=MagicMock(status_code=413, text="Context length exceeded"),
            )
        elif error_type == "rate_limit":
            return httpx.HTTPStatusError(
                message="Rate limit exceeded",
                request=MagicMock(),
                response=MagicMock(status_code=429, text="Too many requests"),
            )
        elif error_type == "invalid_api_key":
            return httpx.HTTPStatusError(
                message="Invalid API Key",
                request=MagicMock(),
                response=MagicMock(status_code=401, text="Invalid API Key"),
            )
        else:
            raise Exception("unknown error type")

    def mock_api_call(self, response_data: Dict[str, Any]) -> MagicMock:
        """Mock HuggingFace chat_completion call on AsyncInferenceClient."""
        content = response_data.get("text", "Hello, world!")

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 15
            total_tokens = 25

        class _Message:
            def __init__(self, content: str):
                self.role = "assistant"
                self.content = content
                self.tool_calls = None

        class _Choice:
            def __init__(self, content: str):
                self.message = _Message(content)
                self.finish_reason = "stop"

        class _Completion:
            def __init__(self, content: str):
                self.choices = [_Choice(content)]
                self.usage = _Usage()

        async def mock_chat_completion(*args, **kwargs):
            return _Completion(content)

        return patch.object(AsyncInferenceClient, "chat_completion", side_effect=mock_chat_completion)  # type: ignore[return-value]

    def mock_streaming_call(self, chunks: List[Dict[str, Any]]):
        """Mock HuggingFace TGI streaming API call."""
        text = "".join(chunk.get("content", "") for chunk in chunks)
        self.create_huggingface_streaming_responses(text)

        class _Delta:
            def __init__(self, content):
                self.content = content
                self.tool_calls = None

        class _Choice:
            def __init__(self, content: str, done: bool):
                self.delta = _Delta(content)
                self.finish_reason = "stop" if done else None

        class _Chunk:
            def __init__(self, content: str, done: bool, is_last: bool):
                self.choices = [_Choice(content, done)]
                # Only the last chunk includes usage
                if is_last:

                    class _Usage:
                        prompt_tokens = 10
                        completion_tokens = len(text.split())
                        total_tokens = 10 + len(text.split())

                    self.usage = _Usage()
                else:
                    self.usage = None

        async def mock_stream():
            words = text.split()
            for i, word in enumerate(words):
                is_last = i == len(words) - 1
                content = word + (" " if not is_last else "")
                yield _Chunk(content, is_last, is_last)

        async def mock_chat_completion(*args, **kwargs):
            return mock_stream()

        return patch.object(AsyncInferenceClient, "chat_completion", side_effect=mock_chat_completion)

    def mock_error_response(self, error_type: str):
        """Mock HuggingFace API error response."""
        error = self.create_huggingface_error(error_type)
        return patch.object(AsyncInferenceClient, "chat_completion", side_effect=error)

    @pytest.mark.asyncio
    async def test_4xx_errors_not_retried(self):
        """Test that 4xx errors are not retried (no sleep calls)."""
        provider = self.create_provider()

        # Test various 4xx error codes
        error_codes = [404, 413, 429, 422]

        for status_code in error_codes:
            # Create an error with the specific status code
            error = httpx.HTTPStatusError(
                message=f"Error {status_code}",
                request=MagicMock(),
                response=MagicMock(status_code=status_code, text=f"Error {status_code}"),
            )

            with (
                patch.object(AsyncInferenceClient, "chat_completion", side_effect=error),
                patch("nodetool.providers.huggingface_provider.asyncio.sleep") as mock_sleep,
                pytest.raises(Exception) as exc_info,
            ):
                await provider.generate_message(self.create_simple_messages(), "test-model")

            # Verify the error was raised
            assert f"{status_code}" in str(exc_info.value)

            # Verify asyncio.sleep was NOT called (no retries)
            assert mock_sleep.call_count == 0, (
                f"4xx error {status_code} should not be retried, but sleep was called {mock_sleep.call_count} times"
            )

    @pytest.mark.asyncio
    async def test_5xx_errors_retried_with_exponential_backoff(self):
        """Test that 5xx errors are retried with exponential backoff."""
        provider = self.create_provider()

        # Create a 500 server error
        error = httpx.HTTPStatusError(
            message="Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500, text="Internal Server Error"),
        )

        with (
            patch.object(AsyncInferenceClient, "chat_completion", side_effect=error),
            patch("nodetool.providers.huggingface_provider.asyncio.sleep") as mock_sleep,
        ):
            with pytest.raises(Exception) as exc_info:
                await provider.generate_message(self.create_simple_messages(), "test-model")

            # Verify the error was raised after all retries
            assert "500" in str(exc_info.value)

            # Verify exponential backoff: 3 retries with delays of 1s, 2s, 4s
            assert mock_sleep.call_count == 3, f"Expected 3 sleep calls (retries), got {mock_sleep.call_count}"

            # Verify the delays match exponential backoff
            expected_delays = [1.0, 2.0, 4.0]
            actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
            assert actual_delays == expected_delays, f"Expected delays {expected_delays}, got {actual_delays}"

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        """Test that retries work and succeed on a later attempt."""
        provider = self.create_provider()

        # Create an error for first attempt, success for second
        call_count = 0

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 15
            total_tokens = 25

        class _Message:
            def __init__(self, content: str):
                self.role = "assistant"
                self.content = content
                self.tool_calls = None

        class _Choice:
            def __init__(self, content: str):
                self.message = _Message(content)
                self.finish_reason = "stop"

        class _Completion:
            def __init__(self, content: str):
                self.choices = [_Choice(content)]
                self.usage = _Usage()

        async def mock_chat_completion_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails with 500
                error = httpx.HTTPStatusError(
                    message="Server Error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500, text="Server Error"),
                )
                raise error
            else:
                # Second call succeeds
                return _Completion("Success after retry")

        with (
            patch.object(AsyncInferenceClient, "chat_completion", side_effect=mock_chat_completion_with_retry),
            patch("nodetool.providers.huggingface_provider.asyncio.sleep") as mock_sleep,
        ):
            response = await provider.generate_message(self.create_simple_messages(), "test-model")

            # Verify we got a successful response
            assert response.role == "assistant"
            assert response.content == "Success after retry"

            # Verify sleep was called once (for the first retry)
            assert mock_sleep.call_count == 1
            assert mock_sleep.call_args_list[0][0][0] == 1.0

    @pytest.mark.asyncio
    async def test_tgi_server_integration(self):
        """Test integration with Text Generation Inference server."""
        provider = self.create_provider()

        with self.mock_api_call(ResponseFixtures.simple_text_response("TGI server response")):
            response = await provider.generate_message(self.create_simple_messages(), "microsoft/DialoGPT-medium")

        assert response.role == "assistant"

    @pytest.mark.asyncio
    async def test_huggingface_hub_models(self):
        """Test various HuggingFace Hub models."""
        provider = self.create_provider()

        models = [
            "microsoft/DialoGPT-medium",
            "EleutherAI/gpt-j-6b",
            "bigscience/bloom-1b7",
            "huggingface/CodeBERTa-small-v1",
        ]

        for model in models:
            with self.mock_api_call(ResponseFixtures.simple_text_response(f"Response from {model}")):
                response = await provider.generate_message(self.create_simple_messages(f"Test {model}"), model)
            assert response.role == "assistant"

    @pytest.mark.asyncio
    async def test_custom_parameters(self):
        """Test custom generation parameters."""
        provider = self.create_provider()

        with self.mock_api_call(ResponseFixtures.simple_text_response("Custom response")) as mock_call:
            await provider.generate_message(self.create_simple_messages(), "test-model")

        # Verify parameters were passed
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_length_handling(self):
        """Test handling of context length limits."""
        provider = self.create_provider()

        with self.mock_error_response("token_limit"):
            with pytest.raises(Exception) as exc_info:
                very_long_text = "This is a very long message. " * 1000
                await provider.generate_message(
                    [
                        Message(
                            role="user",
                            content=[MessageTextContent(text=very_long_text)],
                        )
                    ],
                    "test-model",
                )
            assert "413" in str(exc_info.value) or "context" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limit handling."""
        provider = self.create_provider()

        with self.mock_error_response("rate_limit"):
            with pytest.raises(Exception) as exc_info:
                await provider.generate_message(self.create_simple_messages(), "test-model")
            assert "429" in str(exc_info.value) or "rate" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_model_loading_status(self):
        """Test handling model loading and availability."""
        provider = self.create_provider()

        # Test model not found
        with self.mock_error_response("model_not_found"), pytest.raises(httpx.HTTPStatusError):
            await provider.generate_message(self.create_simple_messages(), "nonexistent/model")

    @pytest.mark.asyncio
    async def test_custom_stopping_criteria(self):
        """Test custom stopping sequences."""
        provider = self.create_provider()

        with self.mock_api_call(ResponseFixtures.simple_text_response("Response with stop")) as mock_call:
            await provider.generate_message(self.create_simple_messages(), "test-model")

        mock_call.assert_called_once()
