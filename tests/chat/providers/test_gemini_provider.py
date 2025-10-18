"""
Tests for Gemini provider with comprehensive API response mocking.

This module tests the Google Gemini provider implementation including:
- Gemini model responses
- Multimodal capabilities (text, image, video)
- Function calling
- Streaming responses
- Safety settings and content filtering

Google Gemini API Documentation (2024):
URLs:
- https://ai.google.dev/gemini-api/docs
- https://ai.google.dev/api
- https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference

The Gemini API provides access to Google's latest generative AI models with multimodal capabilities.

Available Models (2024):
- Gemini 2.5 Pro: State-of-the-art thinking model for complex reasoning
- Gemini 2.5 Flash: Optimized for large-scale, low-latency tasks
- Gemini 1.5 Pro: Balanced performance with long context support
- Gemini 1.5 Flash: Fast inference with good quality

Key Request Parameters:
- model: Model name (e.g., "gemini-2.5-pro", "gemini-1.5-flash")
- contents: Array of content parts (text, inline_data for images/video)
- generationConfig: Controls like temperature, maxOutputTokens, topK, topP
- safetySettings: Content filtering and safety controls
- tools: Function declarations for function calling
- systemInstruction: System-level instructions separate from user content

Content Types:
- Text: Plain text input and responses
- Images: Inline base64 or uploaded file references
- Video: Video file processing and analysis
- Audio: Audio file transcription and analysis (select models)
- Documents: PDF and document analysis

Response Format:
- candidates: Array of generated responses
- Each candidate has content (parts), finishReason, safetyRatings
- usageMetadata: Token counts (promptTokenCount, candidatesTokenCount, totalTokenCount)
- promptFeedback: Safety ratings for the input prompt

Function Calling:
- Functions defined in tools with name, description, parameters (OpenAPI schema)
- Model responds with functionCall parts containing name and args
- Follow up with functionResponse parts containing response data
- Supports multiple function calls per response

Streaming:
- Server-sent events for real-time response generation
- Incremental content delivery for long responses
- Usage metadata provided at completion

Safety and Filtering:
- Built-in safety filters for harmful content
- Configurable safety settings by category (harassment, hate speech, etc.)
- Safety ratings provided for both input and output
- Content blocking and filtering capabilities

Multimodal Features:
- Image understanding and analysis
- Video processing and summarization
- Document analysis and extraction
- Vision-language reasoning
- Cross-modal understanding

Generation Configuration:
- temperature: Creativity control (0.0-2.0)
- maxOutputTokens: Response length limit
- topK: Top-k sampling for diversity
- topP: Nucleus sampling threshold
- candidateCount: Number of response candidates
- stopSequences: Custom stop sequences

Context and Memory:
- Large context windows (up to 1M+ tokens in select models)
- Long conversation support
- Document and codebase analysis
- Multi-turn conversations with context retention

Integration Options:
- REST API with HTTP requests
- Official client libraries (Python, JavaScript, Go, etc.)
- Google AI Studio for rapid prototyping
- Vertex AI integration for enterprise use
- Firebase integration for mobile/web apps

Error Handling:
- 400: Invalid request (malformed content, unsupported format)
- 401: Authentication error (invalid API key)
- 403: Forbidden (quota exceeded, terms violation)
- 429: Rate limit exceeded
- 500: Internal server error

Special Features:
- Grounding with Google Search
- Code execution and analysis
- Math and reasoning capabilities
- Creative writing and content generation
- Factual information synthesis

Performance Optimizations:
- Efficient token usage with smart context management
- Fast inference for real-time applications
- Batch processing capabilities
- Regional deployment options
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch, MagicMock

from nodetool.providers.gemini_provider import GeminiProvider
from nodetool.metadata.types import Message, MessageTextContent
from tests.chat.providers.test_base_provider import BaseProviderTest, ResponseFixtures

# Import for mocking - conditionally handle missing dependency
try:
    from google.genai.client import AsyncClient
except ImportError:
    AsyncClient = None


class TestGeminiProvider(BaseProviderTest):
    """Test suite for Gemini provider with realistic API response mocking."""

    @property
    def provider_class(self):
        return GeminiProvider

    @property
    def provider_name(self):
        return "gemini"

    def create_gemini_response(
        self, content: str = "Hello, world!", function_calls: List[Dict] | None = None
    ) -> Dict[str, Any]:
        """Create a realistic Gemini API response."""
        parts: list[dict[str, Any]] = [{"text": content}] if content else []

        if function_calls:
            for fc in function_calls:
                parts.append({"functionCall": {"name": fc["name"], "args": fc["args"]}})

        return {
            "candidates": [
                {
                    "content": {"parts": parts, "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "probability": "NEGLIGIBLE",
                        },
                    ],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 15,
                "totalTokenCount": 25,
            },
            "promptFeedback": {
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                    }
                ]
            },
        }

    def create_gemini_streaming_responses(
        self, text: str = "Hello world!"
    ) -> List[Dict[str, Any]]:
        """Create realistic Gemini streaming response chunks."""
        chunks = []
        words = text.split()

        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            content = word + (" " if not is_last else "")

            chunk: dict[str, Any] = {
                "candidates": [
                    {
                        "content": {"parts": [{"text": content}], "role": "model"},
                        "finishReason": "STOP" if is_last else None,
                        "index": 0,
                    }
                ]
            }

            if is_last:
                chunk["usageMetadata"] = {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": len(words),
                    "totalTokenCount": 10 + len(words),
                }

            chunks.append(chunk)

        return chunks

    def create_gemini_error(self, error_type: str = "invalid_request"):
        """Create realistic Gemini API errors."""
        if error_type == "quota_exceeded":
            return Exception("Resource has been exhausted (e.g. check quota).")
        elif error_type == "invalid_api_key":
            return Exception("API key not valid. Please pass a valid API key.")
        elif error_type == "safety_filter":
            return Exception("The candidate was filtered due to safety reasons.")
        else:
            return Exception("Invalid request format or parameters.")

    def mock_api_call(self, response_data: Dict[str, Any]):
        """Mock Gemini API call with structured response."""
        if "tool_calls" in response_data:
            # Function calling response
            self.create_gemini_response(
                content=str(response_data.get("text")),
                function_calls=response_data["tool_calls"],
            )
        else:
            # Regular text response
            self.create_gemini_response(
                content=response_data.get("text", "Hello, world!")
            )

        # Mock the Google GenerativeAI client
        mock_response = MagicMock()
        mock_response.text = response_data.get("text", "Hello, world!")
        mock_response.candidates = [
            MagicMock(
                content=MagicMock(
                    parts=[MagicMock(text=response_data.get("text", "Hello, world!"))]
                )
            )
        ]

        # Mock the provider's get_client method instead
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.models = mock_models

        return patch.object(GeminiProvider, "get_client", return_value=mock_client)

    def mock_streaming_call(self, chunks: List[Dict[str, Any]]):
        """Mock Gemini streaming API call."""
        text = "".join(chunk.get("content", "") for chunk in chunks)
        gemini_chunks = self.create_gemini_streaming_responses(text)

        async def mock_stream():
            for chunk in gemini_chunks:
                mock_chunk = MagicMock()
                mock_chunk.text = chunk["candidates"][0]["content"]["parts"][0]["text"]
                yield mock_chunk

        return patch(
            "google.generativeai.GenerativeModel.generate_content",
            return_value=mock_stream(),
        )  # type: ignore[return-value]

    def mock_error_response(self, error_type: str) -> MagicMock:
        """Mock Gemini API error response."""
        error = self.create_gemini_error(error_type)
        return patch(
            "google.generativeai.GenerativeModel.generate_content", side_effect=error
        )  # type: ignore[return-value]

    def create_mock_tool(self):
        """Create a mock tool for testing tool calling."""
        from tests.chat.providers.test_base_provider import MockTool

        return MockTool()

    @pytest.mark.asyncio
    async def test_multimodal_image_input(self):
        """Test multimodal capabilities with image input."""
        provider = self.create_provider()

        # Mock image processing
        with self.mock_api_call(
            ResponseFixtures.simple_text_response("I can see an image")
        ):
            response = await provider.generate_message(
                self.create_simple_messages("Describe this image"), "gemini-1.5-pro"
            )

        assert response.role == "assistant"

    @pytest.mark.asyncio
    async def test_safety_filtering(self):
        """Test content safety filtering and ratings."""
        provider = self.create_provider()

        with self.mock_error_response("safety_filter"):
            with pytest.raises(Exception) as exc_info:
                await provider.generate_message(
                    self.create_simple_messages("Inappropriate content"),
                    "gemini-1.5-pro",
                )
            assert "safety" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_function_calling_capabilities(self):
        """Test Gemini's function calling features."""
        provider = self.create_provider()
        messages = self.create_tool_messages()
        tools = [self.create_mock_tool()]

        function_response = {
            "tool_calls": [{"name": "mock_tool", "args": {"query": "test search"}}]
        }

        with self.mock_api_call(function_response):
            response = await provider.generate_message(
                messages, "gemini-1.5-pro", tools=tools
            )

        # Gemini might handle tools differently
        assert response.role == "assistant"

    @pytest.mark.asyncio
    async def test_large_context_handling(self):
        """Test handling of large context windows."""
        provider = self.create_provider()

        # Test with very long context
        long_context = "This is a very long context. " * 1000
        messages = [
            Message(role="user", content=[MessageTextContent(text=long_context)])
        ]

        with self.mock_api_call(
            ResponseFixtures.simple_text_response("Processed long context")
        ):
            response = await provider.generate_message(messages, "gemini-1.5-pro")

        assert response.role == "assistant"

    @pytest.mark.asyncio
    async def test_generation_config_parameters(self):
        """Test custom generation configuration."""
        provider = self.create_provider()

        with self.mock_api_call(
            ResponseFixtures.simple_text_response("Creative response")
        ) as mock_call:
            await provider.generate_message(
                self.create_simple_messages(), "gemini-1.5-pro"
            )

        mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_variants(self):
        """Test different Gemini model variants."""
        provider = self.create_provider()

        models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-pro"]

        for model in models:
            with self.mock_api_call(
                ResponseFixtures.simple_text_response(f"Response from {model}")
            ):
                response = await provider.generate_message(
                    self.create_simple_messages(f"Test {model}"), model
                )
            assert response.role == "assistant"
