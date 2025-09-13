"""
Additional tests for image utilities, specifically for image_ref_to_base64_jpeg.
"""

import base64
import tempfile
import unittest
from unittest.mock import MagicMock, patch, mock_open

from nodetool.media.image.image_utils import image_ref_to_base64_jpeg
from nodetool.metadata.types import ImageRef


class TestImageUtilsExtra(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image (1x1 red pixel)
        self.test_image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9"
        self.expected_base64_prefix = "/9j/"  # JPEG base64 prefix

    @patch("nodetool.media.image.image_utils.image_data_to_base64_jpeg")
    def test_image_ref_to_base64_jpeg_with_direct_data(self, mock_convert):
        """Test converting ImageRef with direct data."""
        mock_convert.return_value = "mocked_base64_data"

        image_ref = MagicMock()
        image_ref.data = self.test_image_data
        image_ref.uri = None

        result = image_ref_to_base64_jpeg(image_ref)

        self.assertEqual(result, "mocked_base64_data")
        mock_convert.assert_called_once_with(self.test_image_data, (512, 512), 85)

    @patch("nodetool.media.image.image_utils.image_data_to_base64_jpeg")
    def test_image_ref_to_base64_jpeg_with_data_uri(self, mock_convert):
        """Test converting ImageRef with data URI."""
        mock_convert.return_value = "mocked_base64_data"

        # Create a data URI with base64 encoded test image
        data_uri = (
            f"data:image/jpeg;base64,{base64.b64encode(self.test_image_data).decode()}"
        )

        image_ref = MagicMock()
        image_ref.data = None
        image_ref.uri = data_uri

        result = image_ref_to_base64_jpeg(image_ref)

        self.assertEqual(result, "mocked_base64_data")
        mock_convert.assert_called_once()

    @patch("nodetool.media.image.image_utils.image_data_to_base64_jpeg")
    @patch("httpx.get")
    def test_image_ref_to_base64_jpeg_with_http_url(self, mock_get, mock_convert):
        """Test converting ImageRef with HTTP URL."""
        mock_response = MagicMock()
        mock_response.content = self.test_image_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        mock_convert.return_value = "mocked_base64_data"

        image_ref = MagicMock()
        image_ref.data = None
        image_ref.uri = "https://example.com/image.jpg"

        result = image_ref_to_base64_jpeg(image_ref)

        self.assertEqual(result, "mocked_base64_data")
        mock_get.assert_called_once_with(
            "https://example.com/image.jpg", 
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        mock_convert.assert_called_once_with(self.test_image_data, (512, 512), 85)

    @patch("nodetool.media.image.image_utils.image_data_to_base64_jpeg")
    def test_image_ref_to_base64_jpeg_with_file_uri(self, mock_convert):
        """Test converting ImageRef with file URI."""
        mock_convert.return_value = "mocked_base64_data"

        with patch("builtins.open", mock_open(read_data=self.test_image_data)):
            image_ref = MagicMock()
            image_ref.data = None
            image_ref.uri = "file:///path/to/image.jpg"

            result = image_ref_to_base64_jpeg(image_ref)

            self.assertEqual(result, "mocked_base64_data")
            mock_convert.assert_called_once_with(self.test_image_data, (512, 512), 85)

    def test_image_ref_to_base64_jpeg_with_no_data_or_uri(self):
        """Test that ValueError is raised when ImageRef has no data or URI."""
        image_ref = MagicMock()
        image_ref.data = None
        # Remove uri attribute entirely to trigger the error
        del image_ref.uri

        with self.assertRaises(ValueError) as context:
            image_ref_to_base64_jpeg(image_ref)

        self.assertIn("ImageRef has no data or URI", str(context.exception))

    def test_image_ref_to_base64_jpeg_with_unsupported_uri(self):
        """Test that ValueError is raised for unsupported URI schemes."""
        image_ref = MagicMock()
        image_ref.data = None
        image_ref.uri = "ftp://example.com/image.jpg"

        with self.assertRaises(ValueError) as context:
            image_ref_to_base64_jpeg(image_ref)

        self.assertIn("Unsupported URI scheme", str(context.exception))

    @patch("httpx.get")
    def test_image_ref_to_base64_jpeg_with_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        mock_get.side_effect = Exception("Network error")

        image_ref = MagicMock()
        image_ref.data = None
        image_ref.uri = "https://example.com/image.jpg"

        with self.assertRaises(ValueError) as context:
            image_ref_to_base64_jpeg(image_ref)

        self.assertIn("Failed to download image", str(context.exception))

    def test_image_ref_to_base64_jpeg_with_invalid_data_uri(self):
        """Test handling of invalid data URIs."""
        image_ref = MagicMock()
        image_ref.data = None
        image_ref.uri = "data:image/jpeg;base64,invalid_base64"

        with self.assertRaises(ValueError) as context:
            image_ref_to_base64_jpeg(image_ref)

        self.assertIn("Invalid data URI", str(context.exception))

    def test_image_ref_to_base64_jpeg_with_custom_max_size_and_quality(self):
        """Test that custom max_size and quality parameters are passed through."""
        with patch(
            "nodetool.media.image.image_utils.image_data_to_base64_jpeg"
        ) as mock_convert:
            mock_convert.return_value = "mocked_base64_data"

            image_ref = MagicMock()
            image_ref.data = self.test_image_data
            image_ref.uri = None

            result = image_ref_to_base64_jpeg(
                image_ref, max_size=(256, 256), quality=90
            )

            self.assertEqual(result, "mocked_base64_data")
            mock_convert.assert_called_once_with(self.test_image_data, (256, 256), 90)


if __name__ == "__main__":
    unittest.main()
