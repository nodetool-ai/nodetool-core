"""
Tests for MiniMax provider with comprehensive API response mocking.

This module tests the MiniMax provider implementation including:
- Provider initialization with API key
- Model responses
- Tool use functionality
- Streaming responses

MiniMax Anthropic API Documentation:
URL: https://platform.minimaxi.com/docs/api-reference/text-anthropic-api
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ToolUseBlock, Usage

from nodetool.metadata.types import Message, MessageTextContent
from nodetool.providers.minimax_provider import MINIMAX_BASE_URL, MiniMaxProvider
from tests.chat.providers.test_base_provider import BaseProviderTest, ResponseFixtures


class TestMiniMaxProvider(BaseProviderTest):
    """Test suite for MiniMax provider with realistic API response mocking."""

    @property
    def provider_class(self):
        return MiniMaxProvider

    @property
    def provider_name(self):
        return "minimax"

    def create_anthropic_message_response(
        self, content: str = "Hello, world!", tool_uses: List[Dict] | None = None
    ) -> AnthropicMessage:
        """Create a realistic Anthropic Message response."""
        content_blocks = []

        if content:
            content_blocks.append(TextBlock(text=content, type="text"))

        if tool_uses:
            for tool_use in tool_uses:
                content_blocks.append(
                    ToolUseBlock(
                        id=tool_use["id"],
                        name=tool_use["name"],
                        input=tool_use["args"],
                        type="tool_use",
                    )
                )

        return AnthropicMessage(
            id="msg_123",
            content=content_blocks,
            model="MiniMax-Text-01",
            role="assistant",
            stop_reason="end_turn" if not tool_uses else "tool_use",
            stop_sequence=None,
            type="message",
            usage=Usage(input_tokens=10, output_tokens=25),
        )

    def create_anthropic_streaming_responses(self, text: str = "Hello world!", chunk_size: int = 5) -> List[Dict]:
        """Create realistic Anthropic streaming response events."""
        events = []

        # Message start event
        events.append(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "MiniMax-Text-01",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            }
        )

        # Content block start
        events.append(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        )

        # Content block deltas
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i : i + chunk_size]
            events.append(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": chunk_text},
                }
            )

        # Content block stop
        events.append({"type": "content_block_stop", "index": 0})

        # Message stop
        events.append({"type": "message_stop"})

        return events

    def create_anthropic_error(self, error_type: str = "rate_limit"):
        """Create realistic Anthropic API errors."""
        if error_type == "rate_limit":
            return anthropic.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Rate limit exceeded",
                    }
                },
            )
        elif error_type == "context_length":
            return anthropic.BadRequestError(
                message="Input is too long",
                response=MagicMock(status_code=400),
                body={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Input is too long",
                    }
                },
            )
        elif error_type == "invalid_api_key":
            return anthropic.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body={"error": {"type": "authentication_error"}},
            )
        else:
            return anthropic.APIError(message="Unknown error")

    def mock_api_call(self, response_data: Dict[str, Any]):
        """Mock Anthropic API call with structured response."""
        if "tool_calls" in response_data:
            # Tool use response
            tool_uses = [{"id": tc["id"], "name": tc["name"], "args": tc["args"]} for tc in response_data["tool_calls"]]

            anthropic_response = self.create_anthropic_message_response(
                content=response_data.get("text"), tool_uses=tool_uses
            )
        else:
            # Regular text response
            anthropic_response = self.create_anthropic_message_response(
                content=response_data.get("text", "Hello, world!")
            )

        # Mock the async create method
        async def mock_create(*args, **kwargs):
            return anthropic_response

        return patch.object(
            anthropic.resources.messages.AsyncMessages,
            "create",
            side_effect=mock_create,
        )

    def mock_streaming_call(self, chunks: List[Dict[str, Any]]):
        """Mock Anthropic streaming API call."""
        # Convert generic chunks to Anthropic streaming events
        text = "".join(chunk.get("content", "") for chunk in chunks)
        anthropic_events = self.create_anthropic_streaming_responses(text)

        class MockStream:
            def __init__(self, events):
                self.events = events
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.events):
                    raise StopAsyncIteration

                event = self.events[self.index]
                self.index += 1

                # Create mock event object
                mock_event = MagicMock()
                mock_event.type = event["type"]

                if event["type"] == "message_start":
                    mock_event.message = MagicMock()
                    mock_event.message.usage = MagicMock()
                    mock_event.message.usage.input_tokens = 10
                elif event["type"] == "content_block_delta":
                    mock_event.delta = MagicMock()
                    mock_event.delta.text = event["delta"]["text"]
                    mock_event.index = event["index"]

                return mock_event

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return patch.object(
            anthropic.resources.messages.AsyncMessages,
            "stream",
            return_value=MockStream(anthropic_events),
        )

    def mock_error_response(self, error_type: str):
        """Mock Anthropic API error response."""
        error = self.create_anthropic_error(error_type)
        return patch.object(anthropic.resources.messages.AsyncMessages, "create", side_effect=error)


class TestMiniMaxProviderInit:
    """Test suite for MiniMax provider initialization."""

    def test_provider_requires_minimax_api_key(self):
        """Test that provider requires MINIMAX_API_KEY."""
        with pytest.raises(AssertionError, match="MINIMAX_API_KEY is required"):
            MiniMaxProvider(secrets={})

    def test_provider_initializes_with_api_key(self):
        """Test provider initialization with API key."""
        provider = MiniMaxProvider(secrets={"MINIMAX_API_KEY": "test-key"})
        assert provider.api_key == "test-key"
        # Verify the provider was initialized correctly
        assert provider.provider_name == "minimax"
        # Verify _clients dict exists (lazy initialization happens on first get_client call)
        assert hasattr(provider, "_clients")

    def test_get_container_env(self):
        """Test get_container_env returns MINIMAX_API_KEY."""
        provider = MiniMaxProvider(secrets={"MINIMAX_API_KEY": "test-api-key"})
        env_vars = provider.get_container_env(None)  # type: ignore

        assert env_vars == {"MINIMAX_API_KEY": "test-api-key"}

    def test_provider_name(self):
        """Test that provider_name is 'minimax'."""
        provider = MiniMaxProvider(secrets={"MINIMAX_API_KEY": "test-key"})
        assert provider.provider_name == "minimax"

    def test_required_secrets(self):
        """Test that required_secrets returns MINIMAX_API_KEY."""
        assert MiniMaxProvider.required_secrets() == ["MINIMAX_API_KEY"]

    def test_context_length(self):
        """Test context length for MiniMax models."""
        provider = MiniMaxProvider(secrets={"MINIMAX_API_KEY": "test-key"})
        assert provider.get_context_length("MiniMax-Text-01") == 200000

    def test_base_url_constant(self):
        """Test MINIMAX_BASE_URL constant."""
        assert MINIMAX_BASE_URL == "https://api.minimax.io/anthropic"


class TestMiniMaxImageGeneration:
    """Test suite for MiniMax image generation functionality."""

    @pytest.fixture
    def provider(self):
        """Create a MiniMax provider instance for testing."""
        return MiniMaxProvider(secrets={"MINIMAX_API_KEY": "test-api-key"})

    @pytest.mark.asyncio
    async def test_get_available_image_models(self, provider):
        """Test get_available_image_models returns known models."""
        from nodetool.providers.minimax_provider import MINIMAX_IMAGE_MODELS

        models = await provider.get_available_image_models()
        assert len(models) == len(MINIMAX_IMAGE_MODELS)
        assert models[0].id == "image-01"
        assert models[0].name == "MiniMax Image-01"
        assert models[0].provider.value == "minimax"

    @pytest.mark.asyncio
    async def test_get_available_image_models_no_api_key(self):
        """Test get_available_image_models returns empty list without API key."""
        # Create provider with empty api_key
        provider = MiniMaxProvider(secrets={"MINIMAX_API_KEY": ""})
        provider.api_key = ""  # Ensure api_key is empty
        models = await provider.get_available_image_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_text_to_image_empty_prompt(self, provider):
        """Test text_to_image raises ValueError for empty prompt."""
        from nodetool.metadata.types import ImageModel, Provider
        from nodetool.providers.types import TextToImageParams

        params = TextToImageParams(
            model=ImageModel(id="image-01", name="MiniMax Image-01", provider=Provider.MiniMax),
            prompt="",
        )

        with pytest.raises(ValueError, match="prompt cannot be empty"):
            await provider.text_to_image(params)

    @pytest.mark.asyncio
    async def test_text_to_image_no_api_key(self):
        """Test text_to_image raises ValueError without API key."""
        from nodetool.metadata.types import ImageModel, Provider
        from nodetool.providers.types import TextToImageParams

        provider = MiniMaxProvider(secrets={"MINIMAX_API_KEY": ""})
        provider.api_key = ""

        params = TextToImageParams(
            model=ImageModel(id="image-01", name="MiniMax Image-01", provider=Provider.MiniMax),
            prompt="A beautiful sunset",
        )

        with pytest.raises(ValueError, match="MINIMAX_API_KEY is required"):
            await provider.text_to_image(params)

    @pytest.mark.asyncio
    async def test_text_to_image_success(self, provider):
        """Test successful text_to_image generation with mocked API."""
        import base64

        from nodetool.metadata.types import ImageModel, Provider
        from nodetool.providers.types import TextToImageParams

        # Create test image data
        test_image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        test_image_b64 = base64.b64encode(test_image_bytes).decode()

        mock_response = {"data": {"image_base64": [test_image_b64]}}

        params = TextToImageParams(
            model=ImageModel(id="image-01", name="MiniMax Image-01", provider=Provider.MiniMax),
            prompt="A beautiful sunset over the ocean",
            width=1024,
            height=1024,
        )

        # Create proper async mock for aiohttp
        class MockResponse:
            status = 200

            async def json(self):
                return mock_response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        class MockSession:
            def post(self, url, json, headers):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        with (
            patch("aiohttp.ClientSession", return_value=MockSession()),
            patch("aiohttp.ClientTimeout"),
        ):
            result = await provider.text_to_image(params)
            # Verify we get the expected image bytes back
            assert result == test_image_bytes

    def test_calculate_aspect_ratio_square(self, provider):
        """Test aspect ratio calculation for square images."""
        assert provider._calculate_aspect_ratio(1024, 1024) == "1:1"

    def test_calculate_aspect_ratio_16_9(self, provider):
        """Test aspect ratio calculation for 16:9 images."""
        assert provider._calculate_aspect_ratio(1920, 1080) == "16:9"

    def test_calculate_aspect_ratio_9_16(self, provider):
        """Test aspect ratio calculation for 9:16 images."""
        assert provider._calculate_aspect_ratio(1080, 1920) == "9:16"

    def test_calculate_aspect_ratio_4_3(self, provider):
        """Test aspect ratio calculation for 4:3 images."""
        assert provider._calculate_aspect_ratio(1600, 1200) == "4:3"

    def test_calculate_aspect_ratio_3_4(self, provider):
        """Test aspect ratio calculation for 3:4 images."""
        assert provider._calculate_aspect_ratio(1200, 1600) == "3:4"

    def test_calculate_aspect_ratio_unknown(self, provider):
        """Test aspect ratio calculation returns None for non-standard ratios."""
        # 5:8 ratio is not in the common ratios (0.625, which is far from any standard ratio)
        result = provider._calculate_aspect_ratio(500, 800)
        assert result is None

    def test_image_api_url_constant(self):
        """Test MINIMAX_IMAGE_API_URL constant."""
        from nodetool.providers.minimax_provider import MINIMAX_IMAGE_API_URL

        assert MINIMAX_IMAGE_API_URL == "https://api.minimax.io/v1/image_generation"


class TestMiniMaxTTS:
    """Test suite for MiniMax text-to-speech functionality."""

    @pytest.fixture
    def provider(self):
        """Create a MiniMax provider instance for testing."""
        return MiniMaxProvider(secrets={"MINIMAX_API_KEY": "test-api-key"})

    @pytest.mark.asyncio
    async def test_get_available_tts_models(self, provider):
        """Test get_available_tts_models returns known models."""
        from nodetool.providers.minimax_provider import MINIMAX_TTS_MODELS, MINIMAX_TTS_VOICES

        models = await provider.get_available_tts_models()
        assert len(models) == len(MINIMAX_TTS_MODELS)
        assert models[0].id == "speech-2.8-hd"
        assert models[0].name == "MiniMax Speech 2.8 HD"
        assert models[0].provider.value == "minimax"
        assert models[0].voices == MINIMAX_TTS_VOICES

    @pytest.mark.asyncio
    async def test_get_available_tts_models_no_api_key(self):
        """Test get_available_tts_models returns empty list without API key."""
        # Create provider with empty api_key
        provider = MiniMaxProvider(secrets={"MINIMAX_API_KEY": ""})
        provider.api_key = ""  # Ensure api_key is empty
        models = await provider.get_available_tts_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_text_to_speech_empty_text(self, provider):
        """Test text_to_speech raises ValueError for empty text."""
        with pytest.raises(ValueError, match="text must not be empty"):
            async for _ in provider.text_to_speech(text="", model="speech-2.8-hd"):
                pass

    @pytest.mark.asyncio
    async def test_text_to_speech_no_api_key(self):
        """Test text_to_speech raises ValueError without API key."""
        # Create provider with empty api_key
        provider = MiniMaxProvider(secrets={"MINIMAX_API_KEY": ""})
        provider.api_key = ""  # Ensure api_key is empty

        with pytest.raises(ValueError, match="MINIMAX_API_KEY is required"):
            async for _ in provider.text_to_speech(text="Hello world", model="speech-2.8-hd"):
                pass

    @pytest.mark.asyncio
    async def test_text_to_speech_success(self, provider):
        """Test successful text_to_speech generation with mocked API."""
        import numpy as np

        # Create test audio data (10 samples of int16)
        test_audio = np.array([100, 200, 300, 400, 500, -100, -200, -300, -400, -500], dtype=np.int16)
        test_audio_hex = test_audio.tobytes().hex()

        mock_response = {
            "data": {
                "audio": test_audio_hex,
                "status": 2,
            },
            "extra_info": {
                "audio_length": 1000,
                "audio_sample_rate": 24000,
                "audio_size": 20,
                "audio_format": "pcm",
                "audio_channel": 1,
            },
            "trace_id": "test-trace-id",
            "base_resp": {
                "status_code": 0,
                "status_msg": "success",
            },
        }

        # Create proper async mock for aiohttp
        class MockResponse:
            status = 200

            async def json(self):
                return mock_response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        class MockSession:
            def post(self, url, json, headers):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        with (
            patch("aiohttp.ClientSession", return_value=MockSession()),
            patch("aiohttp.ClientTimeout"),
        ):
            chunks = []
            async for chunk in provider.text_to_speech(
                text="Hello world", model="speech-2.8-hd", voice="English_Graceful_Lady"
            ):
                chunks.append(chunk)

            # Verify we get the expected audio data back
            assert len(chunks) == 1
            np.testing.assert_array_equal(chunks[0], test_audio)

    @pytest.mark.asyncio
    async def test_text_to_speech_api_error(self, provider):
        """Test text_to_speech handles API error response."""
        mock_response = {
            "data": {},
            "trace_id": "test-trace-id",
            "base_resp": {
                "status_code": 1004,
                "status_msg": "authentication failed",
            },
        }

        class MockResponse:
            status = 200

            async def json(self):
                return mock_response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        class MockSession:
            def post(self, url, json, headers):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        with (
            patch("aiohttp.ClientSession", return_value=MockSession()),
            patch("aiohttp.ClientTimeout"),
            pytest.raises(RuntimeError, match="authentication failed"),
        ):
            async for _ in provider.text_to_speech(text="Hello world", model="speech-2.8-hd"):
                pass

    @pytest.mark.asyncio
    async def test_text_to_speech_http_error(self, provider):
        """Test text_to_speech handles HTTP error response."""

        class MockResponse:
            status = 500

            async def text(self):
                return "Internal Server Error"

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        class MockSession:
            def post(self, url, json, headers):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        with (
            patch("aiohttp.ClientSession", return_value=MockSession()),
            patch("aiohttp.ClientTimeout"),
            pytest.raises(RuntimeError, match="failed with status 500"),
        ):
            async for _ in provider.text_to_speech(text="Hello world", model="speech-2.8-hd"):
                pass

    @pytest.mark.asyncio
    async def test_text_to_speech_speed_clamping(self, provider):
        """Test that speed is clamped to valid range."""
        import numpy as np

        test_audio = np.array([100, 200], dtype=np.int16)
        test_audio_hex = test_audio.tobytes().hex()

        mock_response = {
            "data": {"audio": test_audio_hex, "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }

        captured_payload = {}

        class MockResponse:
            status = 200

            async def json(self):
                return mock_response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        class MockSession:
            def post(self, url, json, headers):
                captured_payload.update(json)
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        # Test speed above max
        with (
            patch("aiohttp.ClientSession", return_value=MockSession()),
            patch("aiohttp.ClientTimeout"),
        ):
            async for _ in provider.text_to_speech(text="Hello", model="speech-2.8-hd", speed=5.0):
                pass
            assert captured_payload["voice_setting"]["speed"] == 2.0

        # Test speed below min
        captured_payload.clear()
        with (
            patch("aiohttp.ClientSession", return_value=MockSession()),
            patch("aiohttp.ClientTimeout"),
        ):
            async for _ in provider.text_to_speech(text="Hello", model="speech-2.8-hd", speed=0.1):
                pass
            assert captured_payload["voice_setting"]["speed"] == 0.5

    def test_tts_api_url_constant(self):
        """Test MINIMAX_TTS_API_URL constant."""
        from nodetool.providers.minimax_provider import MINIMAX_TTS_API_URL

        assert MINIMAX_TTS_API_URL == "https://api.minimax.io/v1/t2a_v2"

    def test_tts_models_constant(self):
        """Test MINIMAX_TTS_MODELS constant."""
        from nodetool.providers.minimax_provider import MINIMAX_TTS_MODELS

        assert len(MINIMAX_TTS_MODELS) == 4
        model_ids = [m["id"] for m in MINIMAX_TTS_MODELS]
        assert "speech-2.8-hd" in model_ids
        assert "speech-2.8-turbo" in model_ids
        assert "speech-2.6-hd" in model_ids
        assert "speech-2.6-turbo" in model_ids

    def test_tts_voices_constant(self):
        """Test MINIMAX_TTS_VOICES constant."""
        from nodetool.providers.minimax_provider import MINIMAX_TTS_VOICES

        assert len(MINIMAX_TTS_VOICES) > 0
        assert "English_Graceful_Lady" in MINIMAX_TTS_VOICES
        assert "English_Lucky_Robot" in MINIMAX_TTS_VOICES

    # Video generation tests
    def test_video_api_url_constants(self):
        """Test video API URL constants."""
        from nodetool.providers.minimax_provider import (
            MINIMAX_FILE_RETRIEVE_URL,
            MINIMAX_VIDEO_GENERATION_URL,
            MINIMAX_VIDEO_QUERY_URL,
        )

        assert MINIMAX_VIDEO_GENERATION_URL == "https://api.minimax.io/v1/video_generation"
        assert MINIMAX_VIDEO_QUERY_URL == "https://api.minimax.io/v1/query/video_generation"
        assert MINIMAX_FILE_RETRIEVE_URL == "https://api.minimax.io/v1/files/retrieve"

    def test_video_models_constant(self):
        """Test MINIMAX_VIDEO_MODELS constant."""
        from nodetool.providers.minimax_provider import MINIMAX_VIDEO_MODELS

        assert len(MINIMAX_VIDEO_MODELS) == 6

        model_ids = [m.id for m in MINIMAX_VIDEO_MODELS]
        assert "MiniMax-Hailuo-2.3" in model_ids
        assert "MiniMax-Hailuo-02" in model_ids
        assert "T2V-01-Director" in model_ids
        assert "T2V-01" in model_ids
        assert "I2V-01-Director" in model_ids
        assert "I2V-01" in model_ids

        # Check supported tasks
        for model in MINIMAX_VIDEO_MODELS:
            assert len(model.supported_tasks) > 0
            for task in model.supported_tasks:
                assert task in ["text_to_video", "image_to_video"]

    @pytest.mark.asyncio
    async def test_get_available_video_models_success(self, provider):
        """Test getting available video models."""
        models = await provider.get_available_video_models()

        assert len(models) == 6
        model_ids = [m.id for m in models]
        assert "MiniMax-Hailuo-2.3" in model_ids
        assert "I2V-01" in model_ids

    @pytest.mark.asyncio
    async def test_get_available_video_models_no_api_key(self):
        """Test getting available video models without API key."""
        # Create provider with empty API key via a mock
        with patch.object(MiniMaxProvider, "__init__", lambda x, secrets: None):
            provider = MiniMaxProvider(secrets={})
            provider.api_key = ""
            models = await provider.get_available_video_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_text_to_video_empty_prompt(self, provider):
        """Test text_to_video with empty prompt raises ValueError."""
        from nodetool.metadata.types import Provider, VideoModel
        from nodetool.providers.types import TextToVideoParams

        params = TextToVideoParams(
            model=VideoModel(id="MiniMax-Hailuo-2.3", name="Test", provider=Provider.MiniMax),
            prompt="",
        )

        with pytest.raises(ValueError, match="prompt cannot be empty"):
            await provider.text_to_video(params)

    @pytest.mark.asyncio
    async def test_text_to_video_no_api_key(self):
        """Test text_to_video without API key raises ValueError."""
        from nodetool.metadata.types import Provider, VideoModel
        from nodetool.providers.types import TextToVideoParams

        with patch.object(MiniMaxProvider, "__init__", lambda x, secrets: None):
            provider = MiniMaxProvider(secrets={})
            provider.api_key = ""

            params = TextToVideoParams(
                model=VideoModel(id="MiniMax-Hailuo-2.3", name="Test", provider=Provider.MiniMax),
                prompt="A test video",
            )

            with pytest.raises(ValueError, match="MINIMAX_API_KEY is required"):
                await provider.text_to_video(params)

    @pytest.mark.asyncio
    async def test_text_to_video_success(self, provider):
        """Test successful text-to-video generation."""
        from nodetool.metadata.types import Provider, VideoModel
        from nodetool.providers.types import TextToVideoParams

        video_data = b"fake_video_data_mp4"

        # Mock responses for each API call
        task_creation_response = {
            "task_id": "test_task_123",
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }

        task_query_response = {
            "status": "success",
            "file_id": "file_123",
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }

        file_retrieve_response = {
            "file": {"download_url": "https://example.com/video.mp4"},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }

        class MockResponse:
            def __init__(self, status, json_data=None, content=None):
                self.status = status
                self._json = json_data
                self._content = content

            async def json(self):
                return self._json

            async def read(self):
                return self._content

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        call_count = {"post": 0, "get": 0}

        class MockSession:
            def post(self, url, json, headers):
                call_count["post"] += 1
                return MockResponse(200, task_creation_response)

            def get(self, url, headers=None):
                call_count["get"] += 1
                if "query/video_generation" in url:
                    return MockResponse(200, task_query_response)
                elif "files/retrieve" in url:
                    return MockResponse(200, file_retrieve_response)
                else:
                    # Download URL
                    return MockResponse(200, content=video_data)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        params = TextToVideoParams(
            model=VideoModel(id="MiniMax-Hailuo-2.3", name="Test", provider=Provider.MiniMax),
            prompt="A beautiful sunset over mountains",
            resolution="1080P",
        )

        with (
            patch("aiohttp.ClientSession", return_value=MockSession()),
            patch("aiohttp.ClientTimeout"),
        ):
            result = await provider.text_to_video(params)

        assert result == video_data
        assert call_count["post"] == 1
        assert call_count["get"] >= 2  # At least query + retrieve + download

    @pytest.mark.asyncio
    async def test_text_to_video_api_error(self, provider):
        """Test text_to_video handles API errors."""
        from nodetool.metadata.types import Provider, VideoModel
        from nodetool.providers.types import TextToVideoParams

        class MockResponse:
            status = 500

            async def text(self):
                return "Internal Server Error"

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        class MockSession:
            def post(self, url, json, headers):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        params = TextToVideoParams(
            model=VideoModel(id="MiniMax-Hailuo-2.3", name="Test", provider=Provider.MiniMax),
            prompt="A test video",
        )

        with (
            patch("aiohttp.ClientSession", return_value=MockSession()),
            patch("aiohttp.ClientTimeout"),
            pytest.raises(RuntimeError, match="failed with status 500"),
        ):
            await provider.text_to_video(params)

    @pytest.mark.asyncio
    async def test_image_to_video_empty_image(self, provider):
        """Test image_to_video with empty image raises ValueError."""
        from nodetool.metadata.types import Provider, VideoModel
        from nodetool.providers.types import ImageToVideoParams

        params = ImageToVideoParams(
            model=VideoModel(id="I2V-01", name="Test", provider=Provider.MiniMax),
            prompt="Animate this image",
        )

        with pytest.raises(ValueError, match="Input image cannot be empty"):
            await provider.image_to_video(image=b"", params=params)

    @pytest.mark.asyncio
    async def test_image_to_video_success(self, provider):
        """Test successful image-to-video generation."""
        from nodetool.metadata.types import Provider, VideoModel
        from nodetool.providers.types import ImageToVideoParams

        video_data = b"fake_video_data_mp4"
        image_data = b"fake_image_data_png"

        # Mock responses for each API call
        task_creation_response = {
            "task_id": "test_task_456",
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }

        task_query_response = {
            "status": "success",
            "file_id": "file_456",
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }

        file_retrieve_response = {
            "file": {"download_url": "https://example.com/video.mp4"},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }

        class MockResponse:
            def __init__(self, status, json_data=None, content=None):
                self.status = status
                self._json = json_data
                self._content = content

            async def json(self):
                return self._json

            async def read(self):
                return self._content

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        captured_payload = {}

        class MockSession:
            def post(self, url, json, headers):
                captured_payload.update(json)
                return MockResponse(200, task_creation_response)

            def get(self, url, headers=None):
                if "query/video_generation" in url:
                    return MockResponse(200, task_query_response)
                elif "files/retrieve" in url:
                    return MockResponse(200, file_retrieve_response)
                else:
                    return MockResponse(200, content=video_data)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        params = ImageToVideoParams(
            model=VideoModel(id="I2V-01", name="Test", provider=Provider.MiniMax),
            prompt="Make this image come alive",
        )

        with (
            patch("aiohttp.ClientSession", return_value=MockSession()),
            patch("aiohttp.ClientTimeout"),
        ):
            result = await provider.image_to_video(image=image_data, params=params)

        assert result == video_data
        # Verify the image was base64 encoded in the payload
        assert "first_frame_image" in captured_payload
        import base64

        assert captured_payload["first_frame_image"] == base64.b64encode(image_data).decode("utf-8")

    @pytest.mark.asyncio
    async def test_poll_video_task_timeout(self, provider):
        """Test that polling times out correctly."""
        from nodetool.providers.minimax_provider import (
            MINIMAX_VIDEO_QUERY_URL,
        )

        # Mock a response that always returns "processing"
        processing_response = {
            "status": "processing",
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }

        class MockResponse:
            status = 200

            async def json(self):
                return processing_response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        class MockSession:
            def get(self, url, headers=None):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        with (
            patch("aiohttp.ClientSession", return_value=MockSession()),
            patch("aiohttp.ClientTimeout"),
            patch("asyncio.sleep", return_value=None),  # Skip actual waiting
            pytest.raises(TimeoutError, match="timed out"),
        ):
            await provider._poll_video_task("test_task", timeout_s=1, poll_interval=1)

    @pytest.mark.asyncio
    async def test_poll_video_task_failure(self, provider):
        """Test that polling handles task failure."""
        failed_response = {
            "status": "failed",
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }

        class MockResponse:
            status = 200

            async def json(self):
                return failed_response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        class MockSession:
            def get(self, url, headers=None):
                return MockResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        with (
            patch("aiohttp.ClientSession", return_value=MockSession()),
            patch("aiohttp.ClientTimeout"),
            pytest.raises(RuntimeError, match="task failed"),
        ):
            await provider._poll_video_task("test_task", timeout_s=60, poll_interval=1)
