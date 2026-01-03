"""Tests for the KieProvider implementation."""

from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.metadata.types import ImageModel, Provider, VideoModel
from nodetool.providers.types import (
    ImageToImageParams,
    ImageToVideoParams,
    TextToImageParams,
    TextToVideoParams,
)

# Import with error handling since this is an optional provider
try:
    from nodetool.providers.kie_provider import (
        KIE_IMAGE_MODELS,
        KIE_VIDEO_MODELS,
        KieProvider,
    )

    KIE_PROVIDER_AVAILABLE = True
except ImportError:
    KIE_IMAGE_MODELS = []
    KIE_VIDEO_MODELS = []
    KieProvider = None
    KIE_PROVIDER_AVAILABLE = False


class TestKieProvider:
    """Tests for KieProvider class."""

    def test_provider_registration(self):
        """Test that KieProvider is properly registered."""
        from nodetool.providers.base import get_registered_provider

        provider_cls, kwargs = get_registered_provider(Provider.KIE)
        assert provider_cls is KieProvider
        assert kwargs == {}

    def test_required_secrets(self):
        """Test that KieProvider requires the correct secrets."""
        assert KieProvider.required_secrets() == ["KIE_API_KEY"]

    def test_initialization(self):
        """Test KieProvider initialization."""
        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        assert provider.api_key == "test_key"
        assert provider.provider_name == "kie"

    def test_initialization_missing_key(self):
        """Test KieProvider initialization with missing API key."""
        with pytest.raises(ValueError, match="KIE_API_KEY is required"):
            KieProvider(secrets={})

    @pytest.mark.asyncio
    async def test_get_available_image_models(self):
        """Test getting available image models."""
        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        models = await provider.get_available_image_models()

        assert len(models) == len(KIE_IMAGE_MODELS)
        assert all(isinstance(m, ImageModel) for m in models)
        assert all(m.provider == Provider.KIE for m in models)

    @pytest.mark.asyncio
    async def test_get_available_video_models(self):
        """Test getting available video models."""
        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        models = await provider.get_available_video_models()

        assert len(models) == len(KIE_VIDEO_MODELS)
        assert all(isinstance(m, VideoModel) for m in models)
        assert all(m.provider == Provider.KIE for m in models)

    def test_get_capabilities(self):
        """Test that KieProvider reports correct capabilities."""
        from nodetool.providers.base import ProviderCapability

        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        capabilities = provider.get_capabilities()

        expected = {
            ProviderCapability.TEXT_TO_IMAGE,
            ProviderCapability.IMAGE_TO_IMAGE,
            ProviderCapability.TEXT_TO_VIDEO,
            ProviderCapability.IMAGE_TO_VIDEO,
        }
        assert capabilities == expected

    def test_image_models_have_correct_tasks(self):
        """Test that image models have correct supported_tasks."""
        text_to_image_ids = {
            "flux-2/pro-text-to-image",
            "flux-2/flex-text-to-image",
            "seedream/4.5-text-to-image",
            "z-image",
            "google/nano-banana",
            "flux-kontext",
        }
        image_to_image_ids = {
            "flux-2/pro-image-to-image",
            "flux-2/flex-image-to-image",
            "seedream/4.5-edit",
        }

        for model in KIE_IMAGE_MODELS:
            if model.id in text_to_image_ids:
                assert "text_to_image" in model.supported_tasks
            elif model.id in image_to_image_ids:
                assert "image_to_image" in model.supported_tasks

    def test_video_models_have_correct_tasks(self):
        """Test that video models have correct supported_tasks."""
        text_to_video_ids = {
            "kling-2.6/text-to-video",
            "grok-imagine/text-to-video",
            "seedance/v1-lite-text-to-video",
            "hailuo/2-3-text-to-video-pro",
        }
        image_to_video_ids = {
            "kling-2.6/image-to-video",
            "grok-imagine/image-to-video",
            "seedance/v1-lite-image-to-video",
            "hailuo/2-3-image-to-video-pro",
        }

        for model in KIE_VIDEO_MODELS:
            if model.id in text_to_video_ids:
                assert "text_to_video" in model.supported_tasks
            elif model.id in image_to_video_ids:
                assert "image_to_video" in model.supported_tasks


@pytest.mark.skipif(not KIE_PROVIDER_AVAILABLE, reason="KieProvider not available")
class TestKieProviderApiInteraction:
    """Tests for KieProvider API interaction (mocked)."""

    @pytest.mark.asyncio
    async def test_text_to_image_empty_prompt(self):
        """Test text_to_image with empty prompt raises error."""
        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        params = TextToImageParams(
            model=ImageModel(id="flux-2/pro-text-to-image", name="test", provider=Provider.KIE),
            prompt="",
        )

        with pytest.raises(ValueError, match="prompt must not be empty"):
            await provider.text_to_image(params)

    @pytest.mark.asyncio
    async def test_image_to_image_empty_prompt(self):
        """Test image_to_image with empty prompt raises error."""
        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        params = ImageToImageParams(
            model=ImageModel(id="flux-2/pro-image-to-image", name="test", provider=Provider.KIE),
            prompt="",
        )

        with pytest.raises(ValueError, match="prompt must not be empty"):
            await provider.image_to_image(b"fake_image", params)

    @pytest.mark.asyncio
    async def test_text_to_video_empty_prompt(self):
        """Test text_to_video with empty prompt raises error."""
        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        params = TextToVideoParams(
            model=VideoModel(id="kling-2.6/text-to-video", name="test", provider=Provider.KIE),
            prompt="",
        )

        with pytest.raises(ValueError, match="prompt must not be empty"):
            await provider.text_to_video(params)

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_text_to_image_api_call(self, mock_session_class):
        """Test text_to_image makes correct API calls."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock()

        # Mock POST response (submit task)
        mock_post_response = MagicMock()
        mock_post_response.status = 200
        mock_post_response.json = AsyncMock(
            return_value={"code": 200, "message": "success", "data": {"taskId": "task_123"}}
        )
        mock_session.post = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_post_response), __aexit__=AsyncMock())
        )

        # Mock GET responses (poll status and download)
        mock_get_response = MagicMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(
            return_value={
                "code": 200,
                "data": {"state": "success", "resultJson": '{"resultUrls": ["https://example.com/image.png"]}'},
            }
        )

        # Mock download response
        mock_download_response = MagicMock()
        mock_download_response.status = 200
        mock_download_response.read = AsyncMock(return_value=b"fake_image_bytes")

        # Setup mock to return different responses for different URLs
        def mock_get_side_effect(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            if "recordInfo" in url:
                return MagicMock(__aenter__=AsyncMock(return_value=mock_get_response), __aexit__=AsyncMock())
            else:
                return MagicMock(__aenter__=AsyncMock(return_value=mock_download_response), __aexit__=AsyncMock())

        mock_session.get = MagicMock(side_effect=mock_get_side_effect)

        # Execute
        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        params = TextToImageParams(
            model=ImageModel(id="flux-2/pro-text-to-image", name="test", provider=Provider.KIE),
            prompt="a beautiful sunset",
            width=1024,
            height=1024,
        )

        result = await provider.text_to_image(params)

        # Verify
        assert result == b"fake_image_bytes"
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_text_to_image_with_extra_params(self, mock_session_class):
        """Test text_to_image includes extra parameters in API call."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"code": 200, "data": {"taskId": "task_456"}})
        mock_session.post = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock())
        )

        # Execute
        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        params = TextToImageParams(
            model=ImageModel(id="seedream/4.5-text-to-image", name="test", provider=Provider.KIE),
            prompt="sunset",
        )
        # Add extra parameters that are supported by Kie nodes
        params.quality = "high"
        params.aspect_ratio = "16:9"
        params.steps = 30

        # We need to mock _poll_status and _download_result to avoid full execution
        provider._poll_status = AsyncMock()
        provider._download_result = AsyncMock(return_value=b"image")

        await provider.text_to_image(params)

        # Verify post payload
        _, kwargs = mock_session.post.call_args
        payload = kwargs["json"]
        input_params = payload["input"]

        assert input_params["steps"] == 30
        assert input_params["prompt"] == "sunset"

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_text_to_image_with_image_size(self, mock_session_class):
        """Test text_to_image handles Enum parameters like image_size."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"code": 200, "data": {"taskId": "task_789"}})
        mock_session.post = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock())
        )

        # Execute
        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})
        params = TextToImageParams(
            model=ImageModel(id="google/nano-banana", name="test", provider=Provider.KIE),
            prompt="futuristic city",
        )

        # Mock an Enum for image_size
        class MockImageSize(str, Enum):
            SQUARE = "1:1"

        params.image_size = MockImageSize.SQUARE

        # Mock internals
        provider._poll_status = AsyncMock()
        provider._download_result = AsyncMock(return_value=b"image")

        await provider.text_to_image(params)

        # Verify post payload
        _, kwargs = mock_session.post.call_args
        payload = kwargs["json"]
        input_params = payload["input"]

        assert input_params["image_size"] == "1:1"
        assert input_params["prompt"] == "futuristic city"

    @pytest.mark.asyncio
    async def test_check_response_status_error_codes(self):
        """Test that error status codes raise appropriate errors."""
        provider = KieProvider(secrets={"KIE_API_KEY": "test_key"})

        # Test each error code
        error_codes = {
            401: "Unauthorized",
            402: "Insufficient Credits",
            404: "Not Found",
            422: "Validation Error",
            429: "Rate Limited",
            455: "Service Unavailable",
            500: "Server Error",
            501: "Generation Failed",
            505: "Feature Disabled",
        }

        for code, expected_msg in error_codes.items():
            with pytest.raises(ValueError, match=expected_msg):
                provider._check_response_status({"code": code})


@pytest.mark.skipif(not KIE_PROVIDER_AVAILABLE, reason="KieProvider not available")
class TestKieImageModels:
    """Tests for KIE_IMAGE_MODELS."""

    def test_all_models_have_required_fields(self):
        """Test that all image models have required fields."""
        for model in KIE_IMAGE_MODELS:
            assert model.id, f"Model missing id: {model}"
            assert model.name, f"Model missing name: {model}"
            assert model.provider == Provider.KIE, f"Model has wrong provider: {model}"
            assert model.supported_tasks, f"Model missing supported_tasks: {model}"

    def test_model_ids_are_unique(self):
        """Test that all model IDs are unique."""
        ids = [m.id for m in KIE_IMAGE_MODELS]
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"


@pytest.mark.skipif(not KIE_PROVIDER_AVAILABLE, reason="KieProvider not available")
class TestKieVideoModels:
    """Tests for KIE_VIDEO_MODELS."""

    def test_all_models_have_required_fields(self):
        """Test that all video models have required fields."""
        for model in KIE_VIDEO_MODELS:
            assert model.id, f"Model missing id: {model}"
            assert model.name, f"Model missing name: {model}"
            assert model.provider == Provider.KIE, f"Model has wrong provider: {model}"
            assert model.supported_tasks, f"Model missing supported_tasks: {model}"

    def test_model_ids_are_unique(self):
        """Test that all model IDs are unique."""
        ids = [m.id for m in KIE_VIDEO_MODELS]
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"
