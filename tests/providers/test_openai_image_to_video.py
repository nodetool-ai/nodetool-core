"""
Tests for OpenAI image-to-video automatic dimension matching.

This module tests the automatic dimension extraction and snapping functionality
for OpenAI's image-to-video generation.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from nodetool.metadata.types import Provider, VideoModel
from nodetool.providers.openai_provider import OpenAIProvider
from nodetool.providers.types import ImageToVideoParams


class TestOpenAIImageToVideoDimensions:
    """Test suite for automatic dimension matching in image-to-video generation."""

    def create_provider(self) -> OpenAIProvider:
        """Create an OpenAIProvider instance with test secrets.

        Returns:
            An initialized OpenAIProvider with test API key
        """
        return OpenAIProvider(secrets={"OPENAI_API_KEY": "test-key"})

    def create_test_image(self, width: int, height: int, color: str = "red") -> bytes:
        """Create a test image with specified dimensions.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            color: Fill color for the image

        Returns:
            Image data as bytes
        """
        img = Image.new("RGB", (width, height), color=color)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    def test_extract_image_dimensions_landscape(self):
        """Test extracting dimensions from a landscape image."""
        provider = self.create_provider()
        image = self.create_test_image(1920, 1080)

        width, height = provider._extract_image_dimensions(image)

        assert width == 1920
        assert height == 1080

    def test_extract_image_dimensions_portrait(self):
        """Test extracting dimensions from a portrait image."""
        provider = self.create_provider()
        image = self.create_test_image(1080, 1920)

        width, height = provider._extract_image_dimensions(image)

        assert width == 1080
        assert height == 1920

    def test_extract_image_dimensions_square(self):
        """Test extracting dimensions from a square image."""
        provider = self.create_provider()
        image = self.create_test_image(1024, 1024)

        width, height = provider._extract_image_dimensions(image)

        assert width == 1024
        assert height == 1024

    def test_extract_image_dimensions_invalid(self):
        """Test that invalid image data raises ValueError."""
        provider = self.create_provider()
        invalid_image = b"not a valid image"

        with pytest.raises(ValueError, match="Failed to extract image dimensions"):
            provider._extract_image_dimensions(invalid_image)

    def test_snap_to_valid_video_dimensions_16_9_landscape(self):
        """Test snapping 16:9 landscape dimensions."""
        provider = self.create_provider()

        # Test exact match
        result = provider._snap_to_valid_video_dimensions(1280, 720)
        assert result == "1280x720"

        # Test high-res 16:9 (should snap to 1280x720)
        result = provider._snap_to_valid_video_dimensions(1920, 1080)
        assert result == "1280x720"

        # Test slightly different 16:9 (should snap to 1280x720)
        result = provider._snap_to_valid_video_dimensions(1600, 900)
        assert result == "1280x720"

    def test_snap_to_valid_video_dimensions_9_16_portrait(self):
        """Test snapping 9:16 portrait dimensions."""
        provider = self.create_provider()

        # Test exact match
        result = provider._snap_to_valid_video_dimensions(720, 1280)
        assert result == "720x1280"

        # Test high-res 9:16 (should snap to 720x1280)
        result = provider._snap_to_valid_video_dimensions(1080, 1920)
        assert result == "720x1280"

    def test_snap_to_valid_video_dimensions_4_3_aspect_ratio(self):
        """Test snapping 4:3 aspect ratio dimensions."""
        provider = self.create_provider()

        # 4:3 should map to closest 16:9 landscape
        result = provider._snap_to_valid_video_dimensions(800, 600)
        assert result == "1280x720"

        result = provider._snap_to_valid_video_dimensions(1024, 768)
        assert result == "1280x720"

    def test_snap_to_valid_video_dimensions_square(self):
        """Test snapping square (1:1) dimensions.

        Square images should map to the smallest supported size.
        For 1:1 ratio, both landscape and portrait are equally valid,
        but we prefer landscape (720p) over portrait as it's more common.
        """
        provider = self.create_provider()

        result = provider._snap_to_valid_video_dimensions(1024, 1024)
        # Square should map to one of the valid sizes
        # The algorithm will choose based on area similarity
        assert result in ["1280x720", "720x1280"]

    def test_snap_to_valid_video_dimensions_wide_aspect_ratio(self):
        """Test snapping ultra-wide aspect ratios."""
        provider = self.create_provider()

        # 21:9 ultra-wide should map to 1280x720 (landscape)
        result = provider._snap_to_valid_video_dimensions(2560, 1080)
        assert result == "1280x720"

    def test_snap_to_valid_video_dimensions_small_image(self):
        """Test snapping small image dimensions."""
        provider = self.create_provider()

        # Small landscape image
        result = provider._snap_to_valid_video_dimensions(640, 360)
        assert result == "1280x720"

        # Small portrait image
        result = provider._snap_to_valid_video_dimensions(360, 640)
        assert result == "720x1280"

    @pytest.mark.asyncio
    async def test_image_to_video_uses_image_dimensions(self):
        """Test that image_to_video extracts and uses image dimensions."""
        provider = self.create_provider()

        # Create a test image with specific dimensions
        test_image = self.create_test_image(1920, 1080)

        # Create params
        params = ImageToVideoParams(
            model=VideoModel(id="sora-2", name="Sora 2", provider=Provider.OpenAI),
            prompt="Test video generation",
            aspect_ratio="1:1",  # This should be ignored
            resolution="480p",  # This should also be ignored
        )

        # Mock the API calls
        with patch.object(
            provider, "_create_video_job_with_image", new_callable=AsyncMock
        ) as mock_create:
            # Create a mock video response
            mock_video = MagicMock()
            mock_video.id = "test-video-id"
            mock_video.status = "completed"
            mock_video.progress = 100
            mock_video.error = None
            mock_create.return_value = mock_video

            with patch.object(
                provider, "_download_video_content", new_callable=AsyncMock
            ) as mock_download:
                mock_download.return_value = b"fake video content"

                # Call image_to_video
                try:
                    await provider.image_to_video(test_image, params, timeout_s=60)
                except Exception:
                    pass  # We're just checking the call parameters

                # Verify that _create_video_job_with_image was called
                assert mock_create.called

                # Get the call arguments
                call_kwargs = mock_create.call_args[1]

                # Verify that size was derived from image (1920x1080 -> 1280x720)
                assert call_kwargs["size"] == "1280x720"

                # Verify that aspect_ratio and resolution from params were NOT used
                # (they would have resulted in different dimensions)

    @pytest.mark.asyncio
    async def test_image_to_video_preserves_aspect_ratio(self):
        """Test that different aspect ratios are preserved when snapping."""
        provider = self.create_provider()

        test_cases = [
            # (width, height, expected_size)
            (1920, 1080, "1280x720"),  # 16:9 landscape snaps to 1280x720
            (1080, 1920, "720x1280"),  # 9:16 portrait snaps to 720x1280
            (1280, 720, "1280x720"),  # 16:9 landscape exact match
            (720, 1280, "720x1280"),  # 9:16 portrait exact match
        ]

        for width, height, expected_size in test_cases:
            test_image = self.create_test_image(width, height)

            params = ImageToVideoParams(
                model=VideoModel(id="sora-2", name="Sora 2", provider=Provider.OpenAI),
                prompt="Test",
            )

            with patch.object(
                provider, "_create_video_job_with_image", new_callable=AsyncMock
            ) as mock_create:
                mock_video = MagicMock()
                mock_video.id = "test-id"
                mock_video.status = "completed"
                mock_video.progress = 100
                mock_video.error = None
                mock_create.return_value = mock_video

                with patch.object(
                    provider, "_download_video_content", new_callable=AsyncMock
                ) as mock_download:
                    mock_download.return_value = b"fake video"

                    try:
                        await provider.image_to_video(test_image, params, timeout_s=60)
                    except Exception:
                        pass

                    call_kwargs = mock_create.call_args[1]
                    assert (
                        call_kwargs["size"] == expected_size
                    ), f"Expected {expected_size} for {width}x{height}, got {call_kwargs['size']}"

    @pytest.mark.asyncio
    async def test_image_to_video_invalid_image_raises_error(self):
        """Test that invalid image data raises a clear error."""
        provider = self.create_provider()

        invalid_image = b"not a valid image"

        params = ImageToVideoParams(
            model=VideoModel(id="sora-2", name="Sora 2", provider=Provider.OpenAI),
            prompt="Test",
        )

        with pytest.raises(
            ValueError, match="Could not prepare image for video generation"
        ):
            await provider.image_to_video(invalid_image, params, timeout_s=60)
