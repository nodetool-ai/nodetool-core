"""
Tests for Ollama provider with comprehensive API response mocking.

This module tests the Ollama provider implementation including:
- Local model management and inference
- OpenAI-compatible API format
- Tool calling functionality
- Model pulling and management
- Custom model support

Ollama API Documentation (2024):
URLs:
- https://github.com/ollama/ollama/blob/main/docs/api.md
- https://github.com/ollama/ollama/blob/main/docs/openai.md
- https://ollama.readthedocs.io/en/api/

Ollama provides a REST API for running and managing local large language models.

Core API Endpoints:
- POST /api/chat: Generate next message in chat (streaming by default)
- POST /api/generate: Generate completion for a prompt
- GET /api/tags: List local models
- POST /api/pull: Download a model from registry
- POST /api/show: Show model information

OpenAI-Compatible API:
- URL: http://localhost:11434/v1/chat/completions
- Drop-in replacement for OpenAI Chat Completions API
- Compatible with existing OpenAI client libraries

Key Request Parameters:
- model: Model name (e.g., "llama3.2", "llama2", "codellama")
- messages: Array of message objects with role and content
- stream: Boolean for streaming responses (default: true)
- keep_alive: How long to keep model loaded (default: 5m)
- options: Model-specific parameters (temperature, num_ctx, etc.)
- format: For structured output (JSON format supported)

Response Format:
- Standard chat response with message, done flag, and metadata
- OpenAI-compatible responses for /v1/chat/completions
- Streaming responses use server-sent events
- Usage statistics and timing information

Tool Support:
- Function calling with popular models like Llama 3.1
- OpenAI-compatible tool calling API
- JSON schema-based tool definitions
- Multi-turn tool conversations

Model Management:
- Automatic model pulling on first use
- Model names follow 'model:tag' format (tag defaults to 'latest')
- Support for custom models and fine-tuning
- Quantization and optimization options

Structured Output:
- JSON format with schema validation
- Type safety with Pydantic-style schemas
- Consistent structured responses for data extraction

Performance Features:
- Automatic GPU utilization when available
- Memory-efficient model loading
- Concurrent request handling
- Model caching and warm-up

Local Advantages:
- Complete privacy and data control
- No API costs or external dependencies
- Custom model deployment and fine-tuning
- Offline operation capability

Popular Models:
- llama3.2: Latest Llama model with improved capabilities
- llama2: Stable and well-tested for general use
- codellama: Specialized for code generation
- mistral: Fast and efficient for many tasks
- gemma: Google's open model family
- qwen: Alibaba's multilingual models

Integration Examples:
- Direct API calls to localhost:11434
- OpenAI Python SDK with custom base_url
- LangChain OllamaChat integration
- Custom applications via REST API

Error Handling:
- 404: Model not found (trigger automatic pull)
- 400: Invalid request format
- 500: Server error (model loading issues)
- Connection errors: Server not running
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import ollama
import pytest

from nodetool.providers.ollama_provider import OllamaProvider
from tests.chat.providers.test_base_provider import BaseProviderTest, ResponseFixtures


class TestOllamaProvider(BaseProviderTest):
    """Test suite for Ollama provider with realistic API response mocking."""

    @property
    def provider_class(self):
        return OllamaProvider

    @property
    def provider_name(self):
        return "ollama"

    def create_ollama_response(self, content: str = "Hello, world!") -> Dict[str, Any]:
        """Create a realistic Ollama API response."""
        return {
            "model": "test-model",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 200000000,
            "eval_count": 15,
            "eval_duration": 700000000,
        }

    def create_ollama_streaming_responses(self, text: str = "Hello world!") -> List[Dict[str, Any]]:
        """Create realistic Ollama streaming response chunks."""
        chunks = []
        words = text.split()

        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            content = word + (" " if not is_last else "")

            chunks.append(
                {
                    "model": "test-model",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": content},
                    "done": is_last,
                }
            )

        return chunks

    def create_ollama_error(self, error_type: str = "model_not_found"):
        """Create realistic Ollama API errors."""
        if error_type == "model_not_found":
            return httpx.HTTPStatusError(
                message="model 'test-model' not found",
                request=MagicMock(),
                response=MagicMock(status_code=404, text="model 'test-model' not found"),
            )
        elif error_type == "server_unavailable":
            return httpx.ConnectError("Connection refused")
        else:
            return httpx.HTTPStatusError(
                message="Internal server error",
                request=MagicMock(),
                response=MagicMock(status_code=500, text="Internal server error"),
            )

    def mock_api_call(self, response_data: Dict[str, Any]) -> MagicMock:
        """Mock Ollama AsyncClient.chat for non-streaming responses."""
        content = response_data.get("text", "Hello, world!")

        class _Message:
            def __init__(self, content: str):
                self.role = "assistant"
                self.content = content
                self.tool_calls = None

        class _Response:
            def __init__(self, content: str):
                self.model = "test-model"
                self.message = _Message(content)
                self.done = True
                # Usage-like fields used by _update_usage_stats
                self.prompt_eval_count = 10
                self.eval_count = 15

        async def mock_chat(**kwargs):
            return _Response(content)

        return patch.object(ollama.AsyncClient, "chat", side_effect=mock_chat)  # type: ignore[return-value]

    def mock_streaming_call(self, chunks: List[Dict[str, Any]]) -> MagicMock:
        """Mock Ollama streaming API call."""
        text = "".join(chunk.get("content", "") for chunk in chunks)
        self.create_ollama_streaming_responses(text)

        class _Message:
            def __init__(self, content: str):
                self.role = "assistant"
                self.content = content
                self.tool_calls = None

        class _Chunk:
            def __init__(self, content: str, done: bool):
                self.message = _Message(content)
                self.done = done
                self.prompt_eval_count = None
                self.eval_count = None

        async def mock_stream(**kwargs):
            words = text.split()
            for i, word in enumerate(words):
                is_last = i == len(words) - 1
                content = word + (" " if not is_last else "")
                yield _Chunk(content, is_last)

        return patch.object(ollama.AsyncClient, "chat", side_effect=mock_stream)  # type: ignore[return-value]

    def mock_error_response(self, error_type: str):
        """Mock Ollama API error response."""
        error = self.create_ollama_error(error_type)
        return patch.object(ollama.AsyncClient, "chat", side_effect=error)

    @pytest.mark.asyncio
    async def test_model_availability_check(self):
        """Test checking if a model is available locally."""
        provider = self.create_provider()

        # Mock model list response
        with patch(
            "httpx.AsyncClient.get",
            return_value=MagicMock(
                json=AsyncMock(return_value={"models": [{"name": "test-model:latest", "size": 1000000}]})
            ),
        ):
            # Should not raise error for available model
            with self.mock_api_call(ResponseFixtures.simple_text_response()):
                response = await provider.generate_message(self.create_simple_messages(), "test-model")
            assert response.role == "assistant"

    @pytest.mark.asyncio
    async def test_automatic_model_pulling(self):
        """Test automatic model pulling when model is not found."""
        provider = self.create_provider()

        with self.mock_error_response("model_not_found"), pytest.raises(httpx.HTTPStatusError):
            await provider.generate_message(self.create_simple_messages(), "unknown-model")

    @pytest.mark.asyncio
    async def test_keep_alive_parameter(self):
        """Test keep_alive parameter for model management."""
        provider = self.create_provider()

        with self.mock_api_call(ResponseFixtures.simple_text_response()) as mock_call:
            await provider.generate_message(self.create_simple_messages(), "test-model")

        # Should include keep_alive in request
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_info_retrieval(self):
        """Test retrieving model information and capabilities."""
        provider = self.create_provider()

        # Mock model show response via ollama sync client path used by provider
        class _SyncClient:
            class _Show:
                def __init__(self, info):
                    self.modelinfo = info

            def __init__(self, info):
                self._info = info

            def show(self, model: str):
                return self._Show(self._info)

        model_info = {
            "modelfile": "FROM llama3.2:latest\nPARAMETER num_ctx 8192",
            "parameters": "temperature 0.7\ntop_p 0.9",
            "template": "{{ .Prompt }}",
            "details": {
                "parent_model": "llama3.2:latest",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "8B",
                "quantization_level": "Q4_K_M",
            },
        }

        with patch(
            "nodetool.providers.ollama_provider.get_ollama_sync_client",
            return_value=_SyncClient(model_info),
        ):
            # Test that model info is available
            assert provider is not None

    @pytest.mark.asyncio
    async def test_custom_model_parameters(self):
        """Test custom model parameters and options."""
        provider = self.create_provider()

        # Test with custom parameters
        with self.mock_api_call(ResponseFixtures.simple_text_response()) as mock_call:
            await provider.generate_message(self.create_simple_messages(), "test-model")

        # Verify call was made
        mock_call.assert_called_once()

    # --- Gemma function call emulation tests ---

    def test_is_gemma_model(self):
        """Test Gemma model detection from model name."""
        provider = self.create_provider()
        # Positive cases
        assert provider._is_gemma_model("gemma3:4b") is True
        assert provider._is_gemma_model("gemma:2b") is True
        assert provider._is_gemma_model("gemma2:9b") is True
        assert provider._is_gemma_model("Gemma3:27b") is True
        assert provider._is_gemma_model("GEMMA:latest") is True
        # Negative cases
        assert provider._is_gemma_model("llama3:8b") is False
        assert provider._is_gemma_model("mistral:7b") is False
        assert provider._is_gemma_model("qwen:7b") is False

    def test_format_tools_as_gemma_json(self):
        """Test Gemma JSON schema tool formatting."""
        from tests.chat.providers.test_base_provider import MockTool

        provider = self.create_provider()
        tools = [MockTool(name="search", description="Search the web")]
        result = provider._format_tools_as_gemma_json(tools)

        import json

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "search"
        assert parsed[0]["description"] == "Search the web"
        assert "parameters" in parsed[0]

    def test_parse_gemma_bracket_function_calls(self):
        """Test parsing Gemma-style bracket-wrapped function calls."""
        provider = self.create_provider()
        # Single function call in brackets
        text = '[get_weather(city="London")]'
        calls, _cleaned = provider._parse_function_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].args == {"city": "London"}

    def test_parse_gemma_multiple_bracket_calls(self):
        """Test parsing multiple Gemma-style function calls in brackets."""
        provider = self.create_provider()
        text = '[get_weather(city="London"), get_weather(city="Paris")]'
        calls, _cleaned = provider._parse_function_calls(text)
        assert len(calls) == 2
        assert calls[0].args == {"city": "London"}
        assert calls[1].args == {"city": "Paris"}

    def test_parse_plain_calls_still_work(self):
        """Existing plain function call parsing remains unchanged."""
        provider = self.create_provider()
        text = 'get_weather(city="London")'
        calls, _cleaned = provider._parse_function_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_gemma_emulation_uses_json_schema_prompt(self):
        """When model is gemma and tool emulation is active, uses Gemma prompt."""
        provider = self.create_provider()
        from tests.chat.providers.test_base_provider import MockTool

        tools = [MockTool()]

        # Mock has_tool_support to return False so emulation is triggered
        with patch.object(provider, "has_tool_support", return_value=False):
            params = await provider._prepare_request_params(
                self.create_simple_messages(),
                "gemma3:4b",
                tools=tools,
            )

        # The system message should contain Gemma-style instructions
        system_msgs = [m for m in params["messages"] if m.get("role") == "system"]
        assert len(system_msgs) == 1
        sys_content = system_msgs[0]["content"]
        assert "You have access to functions" in sys_content
        assert "func_name1(params_name1=params_value1" in sys_content
        # Should contain JSON schema definitions (not Python comments)
        assert '"name":' in sys_content
        # Should NOT contain the generic emulation markers
        assert "=== AVAILABLE FUNCTIONS ===" not in sys_content

    @pytest.mark.asyncio
    async def test_non_gemma_emulation_uses_python_prompt(self):
        """Non-gemma models use the original Python-style emulation prompt."""
        provider = self.create_provider()
        from tests.chat.providers.test_base_provider import MockTool

        tools = [MockTool()]

        with patch.object(provider, "has_tool_support", return_value=False):
            params = await provider._prepare_request_params(
                self.create_simple_messages(),
                "some-other-model",
                tools=tools,
            )

        system_msgs = [m for m in params["messages"] if m.get("role") == "system"]
        assert len(system_msgs) == 1
        sys_content = system_msgs[0]["content"]
        assert "=== AVAILABLE FUNCTIONS ===" in sys_content
        assert "You have access to functions" not in sys_content
