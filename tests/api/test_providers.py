"""
Tests for Provider API endpoints.
"""

from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nodetool.api.providers import router
from nodetool.api.schemas.common import UsageInfo
from nodetool.api.schemas.requests import (
    AudioSynthesizeRequest,
    AudioTranscribeRequest,
    ChatCompletionRequest,
    ImageGenerateRequest,
    ImageTransformRequest,
    VideoGenerateRequest,
    VideoTransformRequest,
)
from nodetool.api.schemas.responses import (
    ChatCompletionResponse,
    ImageGenerateResponse,
    ProviderInfo,
    ProviderListResponse,
)
from nodetool.metadata.types import (
    ASRModel,
    Chunk,
    ImageModel,
    ImageRef,
    LanguageModel,
    Message,
    Provider,
    ToolCall,
    TTSModel,
    VideoModel,
    VideoRef,
)
from nodetool.providers.base import BaseProvider, ProviderCapability
from nodetool.providers.types import TextToImageParams


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def test_client(app):
    return TestClient(app)


@pytest.fixture
def mock_provider_instance():
    provider = MagicMock(spec=BaseProvider)
    provider.get_capabilities.return_value = {
        ProviderCapability.GENERATE_MESSAGE,
        ProviderCapability.GENERATE_MESSAGES,
        ProviderCapability.TEXT_TO_IMAGE,
        ProviderCapability.IMAGE_TO_IMAGE,
        ProviderCapability.TEXT_TO_SPEECH,
        ProviderCapability.AUTOMATIC_SPEECH_RECOGNITION,
        ProviderCapability.TEXT_TO_VIDEO,
        ProviderCapability.IMAGE_TO_VIDEO,
    }
    provider.generate_message = AsyncMock(return_value=Message(role="assistant", content="Hello!"))
    provider.text_to_image = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n\x00\x00")
    provider.image_to_image = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n\x00\x00")
    provider.automatic_speech_recognition = AsyncMock(return_value="Transcribed text")
    provider.text_to_video = AsyncMock(return_value=b"video_data")
    provider.image_to_video = AsyncMock(return_value=b"video_data")
    return provider


async def async_iter(data):
    """Helper to create async iterator from list."""
    for item in data:
        yield item


class TestProviderListEndpoint:
    """Tests for GET /api/providers/providers"""

    def test_list_providers_returns_valid_response(self, test_client):
        with patch("nodetool.api.providers._get_provider_info_list") as mock_get_providers:
            mock_get_providers.return_value = [
                ProviderInfo(
                    provider=Provider.OpenAI,
                    name="OpenAI",
                    capabilities=[ProviderCapability.GENERATE_MESSAGE],
                    is_available=True,
                )
            ]
            response = test_client.get("/api/providers/providers")
            assert response.status_code == 200
            data = response.json()
            assert "providers" in data
            assert isinstance(data["providers"], list)

    def test_list_providers_empty(self, test_client):
        with patch("nodetool.api.providers._get_provider_info_list") as mock_get_providers:
            mock_get_providers.return_value = []
            response = test_client.get("/api/providers/providers")
            assert response.status_code == 200
            data = response.json()
            assert data["providers"] == []


class TestModelListEndpoints:
    """Tests for model listing endpoints"""

    def test_list_models_endpoint(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_available_models = AsyncMock(
                return_value=[
                    LanguageModel(provider=Provider.OpenAI, id="gpt-4", name="GPT-4"),
                    LanguageModel(provider=Provider.OpenAI, id="gpt-3.5-turbo", name="GPT-3.5 Turbo"),
                ]
            )
            mock_get_provider.return_value = mock_provider

            response = test_client.get("/api/providers/providers/openai/models")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert "pagination" in data

    def test_list_language_models(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_available_language_models = AsyncMock(
                return_value=[
                    LanguageModel(provider=Provider.OpenAI, id="gpt-4", name="GPT-4"),
                ]
            )
            mock_get_provider.return_value = mock_provider

            response = test_client.get("/api/providers/providers/openai/models/language")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1

    def test_list_image_models(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_available_image_models = AsyncMock(
                return_value=[
                    ImageModel(provider=Provider.OpenAI, id="dall-e-3", name="DALL-E 3"),
                ]
            )
            mock_get_provider.return_value = mock_provider

            response = test_client.get("/api/providers/providers/openai/models/image")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_list_tts_models(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_available_tts_models = AsyncMock(
                return_value=[
                    TTSModel(provider=Provider.OpenAI, id="tts-1", name="TTS-1"),
                ]
            )
            mock_get_provider.return_value = mock_provider

            response = test_client.get("/api/providers/providers/openai/models/tts")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_list_asr_models(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_available_asr_models = AsyncMock(
                return_value=[
                    ASRModel(provider=Provider.OpenAI, id="whisper-1", name="Whisper"),
                ]
            )
            mock_get_provider.return_value = mock_provider

            response = test_client.get("/api/providers/providers/openai/models/asr")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_list_video_models(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_available_video_models = AsyncMock(
                return_value=[
                    VideoModel(provider=Provider.Ollama, id="video-gen", name="Video Gen"),
                ]
            )
            mock_get_provider.return_value = mock_provider

            response = test_client.get("/api/providers/providers/ollama/models/video")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)


class TestChatCompletionEndpoint:
    """Tests for POST /api/providers/completions"""

    def test_create_completion_success(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_capabilities.return_value = {ProviderCapability.GENERATE_MESSAGE}
            mock_provider.generate_message = AsyncMock(
                return_value=Message(
                    role="assistant",
                    content="Hello!",
                    usage_prompt_tokens=10,
                    usage_completion_tokens=5,
                    usage_total_tokens=15,
                )
            )
            mock_get_provider.return_value = mock_provider

            response = test_client.post(
                "/api/providers/completions",
                json={
                    "provider": "openai",
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["model"] == "gpt-4"
            assert data["provider"] == "openai"
            assert "message" in data

    def test_create_completion_unsupported_provider(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_capabilities.return_value = set()
            mock_get_provider.return_value = mock_provider

            response = test_client.post(
                "/api/providers/completions",
                json={
                    "provider": "openai",
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert response.status_code == 400
            assert "does not support chat completions" in response.json()["detail"]

    def test_create_completion_invalid_provider(self, test_client):
        response = test_client.post(
            "/api/providers/completions",
            json={
                "provider": "invalid_provider",
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 422


class TestStreamingCompletionEndpoint:
    """Tests for POST /api/providers/completions/stream"""

    def test_streaming_completion_success(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_capabilities.return_value = {ProviderCapability.GENERATE_MESSAGES}

            async def mock_stream():
                yield Chunk(content="Hello")
                yield Chunk(content=" World!")

            mock_provider.generate_messages = mock_stream
            mock_get_provider.return_value = mock_provider

            response = test_client.post(
                "/api/providers/completions/stream",
                json={
                    "provider": "openai",
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")


class TestImageGenerationEndpoint:
    """Tests for POST /api/providers/images/generate"""

    def test_generate_image_success(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_capabilities.return_value = {ProviderCapability.TEXT_TO_IMAGE}
            mock_provider.text_to_image = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n\x00\x00")
            mock_get_provider.return_value = mock_provider

            response = test_client.post(
                "/api/providers/images/generate",
                json={
                    "provider": "openai",
                    "params": {
                        "model": {"provider": "openai", "id": "dall-e-3", "type": "image_model"},
                        "prompt": "A sunset",
                        "width": 1024,
                        "height": 1024,
                    },
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "image" in data
            assert "id" in data


class TestImageTransformEndpoint:
    """Tests for POST /api/providers/images/transform"""

    def test_transform_image_success(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_capabilities.return_value = {ProviderCapability.IMAGE_TO_IMAGE}
            mock_provider.image_to_image = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n\x00\x00")
            mock_get_provider.return_value = mock_provider

            with patch("nodetool.api.providers.fetch_uri_bytes_and_mime", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = ("image/png", b"image_data")

                response = test_client.post(
                    "/api/providers/images/transform",
                    json={
                        "provider": "openai",
                        "image_uri": "data:image/png;base64,abc123",
                        "params": {
                            "model": {"provider": "openai", "id": "dall-e-3", "type": "image_model"},
                            "prompt": "Make it blue",
                            "strength": 0.5,
                        },
                    },
                )
                assert response.status_code == 200
                data = response.json()
                assert "image" in data


class TestAudioSynthesisEndpoint:
    """Tests for POST /api/providers/audio/synthesize"""

    def test_synthesize_audio_success(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_capabilities.return_value = {ProviderCapability.TEXT_TO_SPEECH}

            import numpy as np

            async def mock_audio_stream(text: str, model: str, voice: str | None = None, speed: float = 1.0, timeout_s: int | None = None, **kwargs):
                yield np.array([1, 2, 3], dtype=np.int16)
                yield np.array([4, 5, 6], dtype=np.int16)

            mock_provider.text_to_speech = mock_audio_stream
            mock_get_provider.return_value = mock_provider

            response = test_client.post(
                "/api/providers/audio/synthesize",
                json={
                    "provider": "openai",
                    "model": {"provider": "openai", "id": "tts-1", "type": "tts_model"},
                    "text": "Hello, world!",
                    "voice": "alloy",
                    "speed": 1.0,
                },
            )
            assert response.status_code == 200
            assert "audio" in response.headers.get("content-type", "")


class TestAudioTranscriptionEndpoint:
    """Tests for POST /api/providers/audio/transcribe"""

    def test_transcribe_audio_success(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_capabilities.return_value = {ProviderCapability.AUTOMATIC_SPEECH_RECOGNITION}
            mock_provider.automatic_speech_recognition = AsyncMock(return_value="Hello, world!")
            mock_get_provider.return_value = mock_provider

            with patch("nodetool.api.providers.fetch_uri_bytes_and_mime", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = ("audio/mp3", b"audio_data")

                response = test_client.post(
                    "/api/providers/audio/transcribe",
                    json={
                        "provider": "openai",
                        "model": {"provider": "openai", "id": "whisper-1", "type": "asr_model"},
                        "audio_uri": "data:audio/mp3;base64,abc123",
                        "language": "en",
                    },
                )
                assert response.status_code == 200
                data = response.json()
                assert "text" in data


class TestVideoGenerationEndpoint:
    """Tests for POST /api/providers/video/generate"""

    def test_generate_video_success(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_capabilities.return_value = {ProviderCapability.TEXT_TO_VIDEO}
            mock_provider.text_to_video = AsyncMock(return_value=b"video_data")
            mock_get_provider.return_value = mock_provider

            response = test_client.post(
                "/api/providers/video/generate",
                json={
                    "provider": "ollama",
                    "params": {
                        "model": {"provider": "ollama", "id": "video-gen", "type": "video_model"},
                        "prompt": "A dancing cat",
                        "aspect_ratio": "16:9",
                    },
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "video" in data


class TestVideoTransformEndpoint:
    """Tests for POST /api/providers/video/transform"""

    def test_transform_video_success(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_capabilities.return_value = {ProviderCapability.IMAGE_TO_VIDEO}
            mock_provider.image_to_video = AsyncMock(return_value=b"video_data")
            mock_get_provider.return_value = mock_provider

            with patch("nodetool.api.providers.fetch_uri_bytes_and_mime", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = ("image/png", b"image_data")

                response = test_client.post(
                    "/api/providers/video/transform",
                    json={
                        "provider": "ollama",
                        "image_uri": "data:image/png;base64,abc123",
                        "params": {
                            "model": {"provider": "ollama", "id": "img2vid", "type": "video_model"},
                            "prompt": "Animate this",
                        },
                    },
                )
                assert response.status_code == 200
                data = response.json()
                assert "video" in data


class TestSchemaValidation:
    """Tests for request schema validation"""

    def test_chat_completion_request_validation(self):
        request = ChatCompletionRequest(
            provider=Provider.OpenAI,
            model="gpt-4",
            messages=[Message(role="user", content="Hello")],
        )
        assert request.provider == Provider.OpenAI
        assert request.model == "gpt-4"
        assert len(request.messages) == 1

    def test_image_generate_request_validation(self):
        params = TextToImageParams(
            model=ImageModel(provider=Provider.OpenAI, id="dall-e-3"),
            prompt="A sunset",
            width=1024,
            height=1024,
        )
        request = ImageGenerateRequest(provider=Provider.OpenAI, params=params)
        assert request.provider == Provider.OpenAI
        assert request.params.prompt == "A sunset"

    def test_chat_completion_response_validation(self):
        response = ChatCompletionResponse(
            id="test-123",
            provider=Provider.OpenAI,
            model="gpt-4",
            message=Message(role="assistant", content="Hi!"),
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert response.id == "test-123"
        assert response.message.role == "assistant"
        assert response.usage.total_tokens == 15

    def test_image_generate_response_validation(self):
        response = ImageGenerateResponse(
            id="img-123",
            provider=Provider.OpenAI,
            model="dall-e-3",
            image=ImageRef(uri="data:image/png;base64,abc"),
        )
        assert response.id == "img-123"
        assert "png" in response.image.uri


class TestPagination:
    """Tests for pagination in list endpoints"""

    def test_pagination_calculation(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider_obj = MagicMock()
            mock_provider_obj.get_available_models = AsyncMock(
                return_value=[
                    LanguageModel(provider=Provider.OpenAI, id=f"gpt-{i}", name=f"GPT-{i}")
                    for i in range(25)
                ]
            )
            mock_get_provider.return_value = mock_provider_obj

            response = test_client.get("/api/providers/providers/openai/models?page=1&per_page=10")
            assert response.status_code == 200
            data = response.json()
            assert data["pagination"]["total"] == 25
            assert data["pagination"]["total_pages"] == 3
            assert data["pagination"]["has_next"] is True
            assert data["pagination"]["has_prev"] is False
            assert len(data["models"]) == 10

    def test_pagination_second_page(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider_obj = MagicMock()
            mock_provider_obj.get_available_models = AsyncMock(
                return_value=[
                    LanguageModel(provider=Provider.OpenAI, id=f"gpt-{i}", name=f"GPT-{i}")
                    for i in range(25)
                ]
            )
            mock_get_provider.return_value = mock_provider_obj

            response = test_client.get("/api/providers/providers/openai/models?page=2&per_page=10")
            assert response.status_code == 200
            data = response.json()
            assert data["pagination"]["page"] == 2
            assert data["pagination"]["has_next"] is True
            assert data["pagination"]["has_prev"] is True
            assert len(data["models"]) == 10


class TestErrorHandling:
    """Tests for error handling"""

    def test_invalid_provider_error(self, test_client):
        response = test_client.get("/api/providers/providers/invalid_provider/models")
        assert response.status_code == 422

    def test_unsupported_capability_error(self, test_client):
        with patch("nodetool.api.providers.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_provider.get_capabilities.return_value = set()
            mock_get_provider.return_value = mock_provider

            response = test_client.post(
                "/api/providers/completions",
                json={
                    "provider": "openai",
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert response.status_code == 400
