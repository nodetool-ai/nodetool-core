"""
Tests for API schemas module.
"""

import pytest
from pydantic import ValidationError

from nodetool.api.schemas.common import (
    APIMetadata,
    ErrorDetail,
    ErrorResponse,
    PaginationInfo,
    UsageInfo,
)
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
    AudioTranscribeResponse,
    ChatCompletionChunkEvent,
    ChatCompletionResponse,
    ChatCompletionToolCallEvent,
    ImageGenerateResponse,
    ImageTransformResponse,
    ModelListResponse,
    ProviderInfo,
    ProviderListResponse,
    VideoGenerateResponse,
    VideoTransformResponse,
)
from nodetool.metadata.types import (
    ASRModel,
    Chunk,
    ImageModel,
    ImageRef,
    Message,
    Provider,
    ToolCall,
    TTSModel,
    VideoModel,
    VideoRef,
)
from nodetool.providers.base import ProviderCapability
from nodetool.providers.types import (
    ImageToImageParams,
    ImageToVideoParams,
    TextToImageParams,
    TextToVideoParams,
)


class TestCommonSchemas:
    """Tests for common schema types."""

    def test_usage_info_defaults(self):
        """Test UsageInfo with default values."""
        usage = UsageInfo()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cached_tokens is None
        assert usage.reasoning_tokens is None

    def test_usage_info_with_values(self):
        """Test UsageInfo with custom values."""
        usage = UsageInfo(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            cached_tokens=50,
            reasoning_tokens=25,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 200
        assert usage.total_tokens == 300
        assert usage.cached_tokens == 50
        assert usage.reasoning_tokens == 25

    def test_error_detail(self):
        """Test ErrorDetail schema."""
        error = ErrorDetail(loc=["body", "messages"], msg="Field required", type="missing")
        assert error.loc == ["body", "messages"]
        assert error.msg == "Field required"
        assert error.type == "missing"

    def test_error_response(self):
        """Test ErrorResponse schema."""
        response = ErrorResponse(
            error="validation_error",
            message="Request validation failed",
            details=[ErrorDetail(loc=["body"], msg="Invalid", type="type_error")],
            request_id="req-123",
        )
        assert response.error == "validation_error"
        assert response.message == "Request validation failed"
        assert len(response.details) == 1
        assert response.request_id == "req-123"

    def test_pagination_info(self):
        """Test PaginationInfo schema."""
        pagination = PaginationInfo(page=2, per_page=20, total=100, total_pages=5, has_next=True, has_prev=True)
        assert pagination.page == 2
        assert pagination.per_page == 20
        assert pagination.total == 100
        assert pagination.total_pages == 5
        assert pagination.has_next is True
        assert pagination.has_prev is True

    def test_api_metadata(self):
        """Test APIMetadata schema."""
        metadata = APIMetadata(
            request_id="req-456",
            user_id="user-123",
            workflow_id="wf-789",
            node_id="node-abc",
            extra={"key": "value"},
        )
        assert metadata.request_id == "req-456"
        assert metadata.user_id == "user-123"
        assert metadata.workflow_id == "wf-789"
        assert metadata.node_id == "node-abc"
        assert metadata.extra == {"key": "value"}


class TestRequestSchemas:
    """Tests for request schema types."""

    def test_chat_completion_request_minimal(self):
        """Test ChatCompletionRequest with minimal required fields."""
        request = ChatCompletionRequest(
            provider=Provider.OpenAI, model="gpt-4", messages=[Message(role="user", content="Hello")]
        )
        assert request.provider == Provider.OpenAI
        assert request.model == "gpt-4"
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
        assert request.messages[0].content == "Hello"
        assert request.max_tokens == 8192
        assert request.temperature is None
        assert request.tools is None

    def test_chat_completion_request_full(self):
        """Test ChatCompletionRequest with all fields."""
        request = ChatCompletionRequest(
            provider=Provider.Anthropic,
            model="claude-3-opus",
            messages=[Message(role="system", content="You are helpful"), Message(role="user", content="Hi")],
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            response_format={"type": "json_object"},
            metadata=APIMetadata(request_id="test"),
        )
        assert request.provider == Provider.Anthropic
        assert request.model == "claude-3-opus"
        assert len(request.messages) == 2
        assert request.max_tokens == 4096
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.response_format == {"type": "json_object"}
        assert request.metadata.request_id == "test"

    def test_chat_completion_request_temperature_validation(self):
        """Test temperature validation bounds."""
        # Valid temperature
        request = ChatCompletionRequest(
            provider=Provider.OpenAI, model="gpt-4", messages=[Message(role="user", content="Hi")], temperature=1.5
        )
        assert request.temperature == 1.5

        # Invalid temperature (above max)
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                provider=Provider.OpenAI, model="gpt-4", messages=[Message(role="user", content="Hi")], temperature=3.0
            )

    def test_image_generate_request(self):
        """Test ImageGenerateRequest schema."""
        params = TextToImageParams(
            model=ImageModel(provider=Provider.OpenAI, id="dall-e-3"), prompt="A sunset", width=1024, height=1024
        )
        request = ImageGenerateRequest(provider=Provider.OpenAI, params=params, timeout_s=60)
        assert request.provider == Provider.OpenAI
        assert request.params.prompt == "A sunset"
        assert request.timeout_s == 60

    def test_image_transform_request(self):
        """Test ImageTransformRequest schema."""
        params = ImageToImageParams(
            model=ImageModel(provider=Provider.OpenAI, id="dall-e-3"), prompt="Make it blue", strength=0.5
        )
        request = ImageTransformRequest(
            provider=Provider.OpenAI, image_uri="data:image/png;base64,abc123", params=params
        )
        assert request.provider == Provider.OpenAI
        assert request.image_uri == "data:image/png;base64,abc123"
        assert request.params.prompt == "Make it blue"

    def test_audio_synthesize_request(self):
        """Test AudioSynthesizeRequest schema."""
        model = TTSModel(provider=Provider.OpenAI, id="tts-1", name="TTS-1")
        request = AudioSynthesizeRequest(
            provider=Provider.OpenAI, model=model, text="Hello world", voice="alloy", speed=1.2
        )
        assert request.provider == Provider.OpenAI
        assert request.text == "Hello world"
        assert request.voice == "alloy"
        assert request.speed == 1.2

    def test_audio_transcribe_request(self):
        """Test AudioTranscribeRequest schema."""
        model = ASRModel(provider=Provider.OpenAI, id="whisper-1", name="Whisper")
        request = AudioTranscribeRequest(
            provider=Provider.OpenAI,
            model=model,
            audio_uri="file:///path/to/audio.mp3",
            language="en",
            temperature=0.2,
        )
        assert request.provider == Provider.OpenAI
        assert request.audio_uri == "file:///path/to/audio.mp3"
        assert request.language == "en"
        assert request.temperature == 0.2

    def test_video_generate_request(self):
        """Test VideoGenerateRequest schema."""
        params = TextToVideoParams(
            model=VideoModel(provider=Provider.FalAI, id="video-gen"), prompt="A dancing cat", aspect_ratio="16:9"
        )
        request = VideoGenerateRequest(provider=Provider.FalAI, params=params)
        assert request.provider == Provider.FalAI
        assert request.params.prompt == "A dancing cat"

    def test_video_transform_request(self):
        """Test VideoTransformRequest schema."""
        params = ImageToVideoParams(model=VideoModel(provider=Provider.FalAI, id="img2vid"), prompt="Animate this")
        request = VideoTransformRequest(
            provider=Provider.FalAI, image_uri="asset://image-123", params=params, timeout_s=120
        )
        assert request.provider == Provider.FalAI
        assert request.image_uri == "asset://image-123"
        assert request.timeout_s == 120


class TestResponseSchemas:
    """Tests for response schema types."""

    def test_provider_info(self):
        """Test ProviderInfo schema."""
        info = ProviderInfo(
            provider=Provider.OpenAI,
            name="OpenAI",
            capabilities=[ProviderCapability.GENERATE_MESSAGE, ProviderCapability.TEXT_TO_IMAGE],
            is_available=True,
        )
        assert info.provider == Provider.OpenAI
        assert info.name == "OpenAI"
        assert ProviderCapability.GENERATE_MESSAGE in info.capabilities
        assert info.is_available is True

    def test_provider_list_response(self):
        """Test ProviderListResponse schema."""
        response = ProviderListResponse(
            providers=[
                ProviderInfo(
                    provider=Provider.OpenAI, name="OpenAI", capabilities=[ProviderCapability.GENERATE_MESSAGE]
                ),
                ProviderInfo(
                    provider=Provider.Anthropic, name="Anthropic", capabilities=[ProviderCapability.GENERATE_MESSAGE]
                ),
            ]
        )
        assert len(response.providers) == 2
        assert response.providers[0].provider == Provider.OpenAI

    def test_model_list_response(self):
        """Test ModelListResponse schema."""
        from nodetool.metadata.types import LanguageModel

        response = ModelListResponse(
            models=[LanguageModel(provider=Provider.OpenAI, id="gpt-4", name="GPT-4")],
            pagination=PaginationInfo(total=1, total_pages=1),
        )
        assert len(response.models) == 1
        assert response.pagination.total == 1

    def test_chat_completion_response(self):
        """Test ChatCompletionResponse schema."""
        response = ChatCompletionResponse(
            id="completion-123",
            provider=Provider.OpenAI,
            model="gpt-4",
            message=Message(role="assistant", content="Hello!"),
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=0.001,
        )
        assert response.id == "completion-123"
        assert response.provider == Provider.OpenAI
        assert response.model == "gpt-4"
        assert response.message.role == "assistant"
        assert response.usage.total_tokens == 15
        assert response.cost == 0.001
        assert response.created > 0

    def test_chat_completion_chunk_event(self):
        """Test ChatCompletionChunkEvent schema."""
        event = ChatCompletionChunkEvent(chunk=Chunk(content="Hello"))
        assert event.event == "chunk"
        assert event.chunk.content == "Hello"

    def test_chat_completion_tool_call_event(self):
        """Test ChatCompletionToolCallEvent schema."""
        event = ChatCompletionToolCallEvent(tool_call=ToolCall(id="call-1", name="search", args={"query": "test"}))
        assert event.event == "tool_call"
        assert event.tool_call.name == "search"
        assert event.tool_call.args == {"query": "test"}

    def test_image_generate_response(self):
        """Test ImageGenerateResponse schema."""
        response = ImageGenerateResponse(
            id="img-123",
            provider=Provider.OpenAI,
            model="dall-e-3",
            image=ImageRef(uri="data:image/png;base64,abc"),
            cost=0.04,
        )
        assert response.id == "img-123"
        assert response.provider == Provider.OpenAI
        assert response.image.uri == "data:image/png;base64,abc"
        assert response.cost == 0.04

    def test_image_transform_response(self):
        """Test ImageTransformResponse schema."""
        response = ImageTransformResponse(
            id="transform-456",
            provider=Provider.OpenAI,
            model="dall-e-3",
            image=ImageRef(uri="asset://new-image"),
        )
        assert response.id == "transform-456"
        assert response.image.uri == "asset://new-image"

    def test_audio_transcribe_response(self):
        """Test AudioTranscribeResponse schema."""
        response = AudioTranscribeResponse(
            id="transcribe-789",
            provider=Provider.OpenAI,
            model="whisper-1",
            text="Hello, how are you?",
            language="en",
            duration=5.5,
            cost=0.006,
        )
        assert response.id == "transcribe-789"
        assert response.text == "Hello, how are you?"
        assert response.language == "en"
        assert response.duration == 5.5

    def test_video_generate_response(self):
        """Test VideoGenerateResponse schema."""
        response = VideoGenerateResponse(
            id="video-abc",
            provider=Provider.FalAI,
            model="video-gen",
            video=VideoRef(uri="asset://video-123"),
            duration=10.0,
            cost=0.50,
        )
        assert response.id == "video-abc"
        assert response.video.uri == "asset://video-123"
        assert response.duration == 10.0

    def test_video_transform_response(self):
        """Test VideoTransformResponse schema."""
        response = VideoTransformResponse(
            id="vid-transform-def",
            provider=Provider.FalAI,
            model="img2vid",
            video=VideoRef(uri="asset://animated-video"),
            duration=5.0,
        )
        assert response.id == "vid-transform-def"
        assert response.video.uri == "asset://animated-video"
        assert response.duration == 5.0


class TestTypeReuse:
    """Tests to verify proper reuse of existing types."""

    def test_message_type_reuse_in_request(self):
        """Verify ChatCompletionRequest uses existing Message type."""
        from nodetool.metadata.types import Message as OriginalMessage

        request = ChatCompletionRequest(
            provider=Provider.OpenAI, model="gpt-4", messages=[OriginalMessage(role="user", content="Test")]
        )
        # The message should be the same type
        assert isinstance(request.messages[0], OriginalMessage)

    def test_provider_enum_reuse(self):
        """Verify Provider enum is reused from metadata.types."""
        from nodetool.metadata.types import Provider as OriginalProvider

        request = ChatCompletionRequest(
            provider=OriginalProvider.OpenAI, model="gpt-4", messages=[Message(role="user", content="Test")]
        )
        assert request.provider == OriginalProvider.OpenAI

    def test_image_ref_reuse_in_response(self):
        """Verify ImageRef is reused from metadata.types."""
        from nodetool.metadata.types import ImageRef as OriginalImageRef

        response = ImageGenerateResponse(
            id="test",
            provider=Provider.OpenAI,
            model="dall-e-3",
            image=OriginalImageRef(uri="test://image"),
        )
        assert isinstance(response.image, OriginalImageRef)


class TestJsonSerialization:
    """Tests for JSON serialization of schemas."""

    def test_request_to_json(self):
        """Test serializing request to JSON."""
        request = ChatCompletionRequest(
            provider=Provider.OpenAI, model="gpt-4", messages=[Message(role="user", content="Hello")]
        )
        json_data = request.model_dump_json()
        assert "openai" in json_data.lower()
        assert "gpt-4" in json_data
        assert "Hello" in json_data

    def test_response_to_json(self):
        """Test serializing response to JSON."""
        response = ChatCompletionResponse(
            id="test-123",
            provider=Provider.OpenAI,
            model="gpt-4",
            message=Message(role="assistant", content="Hi!"),
        )
        json_data = response.model_dump_json()
        assert "test-123" in json_data
        assert "Hi!" in json_data

    def test_request_from_json(self):
        """Test deserializing request from dict."""
        data = {
            "provider": "openai",
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
        }
        request = ChatCompletionRequest(**data)
        assert request.provider == Provider.OpenAI
        assert request.model == "gpt-4"
