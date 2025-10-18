"""
Tests for JPEG/base64 image utility functions.
"""

import base64
from io import BytesIO

from PIL import Image

from nodetool.media.image.image_utils import (
    image_data_to_base64_jpeg,
    pil_image_to_base64_jpeg,
)


class TestImageDataToBase64Jpeg:
    """Test the image_data_to_base64_jpeg function."""

    def test_image_data_to_base64_jpeg_basic(self):
        """Test basic image data to base64 JPEG conversion."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_data = img_bytes.getvalue()

        # Convert to base64 JPEG
        result = image_data_to_base64_jpeg(img_data)

        # Verify result is a string
        assert isinstance(result, str)

        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

        # Verify we can decode it back to an image
        decoded_img = Image.open(BytesIO(decoded))
        assert decoded_img.format == "JPEG"
        assert decoded_img.mode == "RGB"

    def test_image_data_to_base64_jpeg_with_resize(self):
        """Test image data conversion with resizing."""
        # Create a large test image
        img = Image.new("RGB", (1000, 1000), color="blue")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_data = img_bytes.getvalue()

        # Convert with size limit
        result = image_data_to_base64_jpeg(img_data, max_size=(512, 512))

        # Decode and verify size
        decoded = base64.b64decode(result)
        decoded_img = Image.open(BytesIO(decoded))

        # Should be resized to fit within 512x512
        assert decoded_img.width <= 512
        assert decoded_img.height <= 512

    def test_image_data_to_base64_jpeg_with_quality(self):
        """Test image data conversion with different quality settings."""
        # Create a test image
        img = Image.new("RGB", (100, 100), color="green")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_data = img_bytes.getvalue()

        # Convert with high quality
        result_high = image_data_to_base64_jpeg(img_data, quality=95)

        # Convert with low quality
        result_low = image_data_to_base64_jpeg(img_data, quality=50)

        # High quality should produce larger base64 string (generally)
        assert len(result_high) >= len(result_low)


class TestPilImageToBase64Jpeg:
    """Test the pil_image_to_base64_jpeg function."""

    def test_pil_image_to_base64_jpeg_basic(self):
        """Test basic PIL image to base64 JPEG conversion."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")

        # Convert to base64 JPEG
        result = pil_image_to_base64_jpeg(img)

        # Verify result is a string
        assert isinstance(result, str)

        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

        # Verify we can decode it back to an image
        decoded_img = Image.open(BytesIO(decoded))
        assert decoded_img.format == "JPEG"
        assert decoded_img.mode == "RGB"

    def test_pil_image_to_base64_jpeg_rgba_to_rgb(self):
        """Test conversion of RGBA image to RGB JPEG."""
        # Create an RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))

        # Convert to base64 JPEG
        result = pil_image_to_base64_jpeg(img)

        # Decode and verify it's RGB
        decoded = base64.b64decode(result)
        decoded_img = Image.open(BytesIO(decoded))
        assert decoded_img.mode == "RGB"

    def test_pil_image_to_base64_jpeg_resize(self):
        """Test PIL image conversion with resizing."""
        # Create a large image
        img = Image.new("RGB", (1000, 1000), color="blue")

        # Convert with size limit
        result = pil_image_to_base64_jpeg(img, max_size=(512, 512))

        # Decode and verify size
        decoded = base64.b64decode(result)
        decoded_img = Image.open(BytesIO(decoded))

        # Should be resized to fit within 512x512
        assert decoded_img.width <= 512
        assert decoded_img.height <= 512

    def test_pil_image_to_base64_jpeg_quality(self):
        """Test PIL image conversion with different quality settings."""
        # Create a test image
        img = Image.new("RGB", (100, 100), color="green")

        # Convert with high quality
        result_high = pil_image_to_base64_jpeg(img, quality=95)

        # Convert with low quality
        result_low = pil_image_to_base64_jpeg(img, quality=50)

        # High quality should produce larger base64 string (generally)
        assert len(result_high) >= len(result_low)
