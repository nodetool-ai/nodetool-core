"""Tests for io.media_fetch module."""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import PIL.Image
import pytest

from nodetool.io.media_fetch import (
    _extract_storage_key_from_url,
    _fetch_file_uri,
    _is_local_storage_url,
    _normalize_image_like_to_png_bytes,
    _parse_asset_id_from_uri,
    _parse_data_uri,
    fetch_uri_bytes_and_mime_async,
)


class TestParseDataUri:
    """Test _parse_data_uri function."""

    def test_simple_base64_data_uri(self) -> None:
        """Test parsing simple base64 data URI."""
        data = b"Hello, world!"
        b64_data = base64.b64encode(data).decode("utf-8")
        uri = f"data:text/plain;base64,{b64_data}"

        mime, result = _parse_data_uri(uri)

        assert mime == "text/plain"
        assert result == data

    def test_data_uri_with_mime_type(self) -> None:
        """Test data URI with explicit MIME type."""
        data = b"test"
        b64_data = base64.b64encode(data).decode("utf-8")
        uri = f"data:image/png;base64,{b64_data}"

        mime, result = _parse_data_uri(uri)

        assert mime == "image/png"
        assert result == data

    def test_data_uri_with_charset(self) -> None:
        """Test data URI with charset parameter."""
        data = b"test"
        b64_data = base64.b64encode(data).decode("utf-8")
        uri = f"data:text/plain;charset=utf-8;base64,{b64_data}"

        mime, result = _parse_data_uri(uri)

        assert mime == "text/plain"
        assert result == data

    def test_data_uri_default_mime_type(self) -> None:
        """Test data URI without explicit MIME type."""
        data = b"test"
        b64_data = base64.b64encode(data).decode("utf-8")
        uri = f"data:;base64,{b64_data}"

        mime, result = _parse_data_uri(uri)

        assert mime == "application/octet-stream"
        assert result == data

    def test_invalid_data_uri_missing_comma(self) -> None:
        """Test that invalid data URI raises ValueError."""
        uri = "data:text/plain"

        with pytest.raises(ValueError, match="Invalid data URI"):
            _parse_data_uri(uri)

    def test_invalid_base64_data(self) -> None:
        """Test that invalid base64 data raises ValueError."""
        uri = "data:text/plain;base64,invalid-base64!@#$"

        with pytest.raises(ValueError, match="Invalid data URI"):
            _parse_data_uri(uri)


class TestFetchFileUri:
    """Test _fetch_file_uri function."""

    @patch("builtins.open")
    def test_fetch_file_uri_success(self, mock_open: Mock) -> None:
        """Test successful file URI fetch."""
        test_data = b"file content"
        mock_file = MagicMock()
        mock_file.read.return_value = test_data
        mock_open.return_value.__enter__.return_value = mock_file

        mime, data = _fetch_file_uri("file:///path/to/file.txt")

        assert data == test_data
        assert mime == "text/plain"
        mock_open.assert_called_once()

    @patch("builtins.open")
    def test_fetch_file_uri_unknown_extension(self, mock_open: Mock) -> None:
        """Test file URI with unknown extension."""
        test_data = b"file content"
        mock_file = MagicMock()
        mock_file.read.return_value = test_data
        mock_open.return_value.__enter__.return_value = mock_file

        mime, data = _fetch_file_uri("file:///path/to/file.unknown")

        assert data == test_data
        assert mime == "application/octet-stream"


class TestNormalizeImageLikeToPngBytes:
    """Test _normalize_image_like_to_png_bytes function."""

    def test_pil_image_conversion(self) -> None:
        """Test converting PIL Image to PNG bytes."""
        img = PIL.Image.new("RGB", (10, 10), color="red")

        result = _normalize_image_like_to_png_bytes(img)

        assert isinstance(result, bytes)
        assert len(result) > 0
        # Verify it's a PNG by checking signature
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_numpy_array_conversion(self) -> None:
        """Test converting numpy array to PNG bytes."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red channel

        result = _normalize_image_like_to_png_bytes(arr)

        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_bytes_conversion(self) -> None:
        """Test converting image bytes to PNG bytes."""
        # Create a small PNG image
        img = PIL.Image.new("RGB", (10, 10), color="blue")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        result = _normalize_image_like_to_png_bytes(img_bytes)

        assert isinstance(result, bytes)
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_invalid_bytes_raises_error(self) -> None:
        """Test that invalid image bytes raises ValueError."""
        invalid_bytes = b"not an image"

        with pytest.raises(ValueError, match="Bytes are not a decodable image"):
            _normalize_image_like_to_png_bytes(invalid_bytes)

    def test_file_like_object_conversion(self) -> None:
        """Test converting file-like object to PNG bytes."""
        img = PIL.Image.new("RGB", (10, 10), color="green")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        result = _normalize_image_like_to_png_bytes(buffer)

        assert isinstance(result, bytes)
        assert result[:8] == b"\x89PNG\r\n\x1a\n"
        # Verify the file position is preserved
        assert buffer.tell() == 0

    def test_unsupported_type_raises_error(self) -> None:
        """Test that unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported object type"):
            _normalize_image_like_to_png_bytes({"not": "an image"})


class TestIsLocalStorageUrl:
    """Test _is_local_storage_url function."""

    def test_localhost_with_port(self) -> None:
        """Test localhost URL with port."""
        url = "http://localhost:7777/api/storage/test.png"
        assert _is_local_storage_url(url) is True

    def test_localhost_without_port(self) -> None:
        """Test localhost URL without port."""
        url = "http://localhost/api/storage/test.png"
        assert _is_local_storage_url(url) is True

    def test_127_0_0_1_with_port(self) -> None:
        """Test 127.0.0.1 URL with port."""
        url = "https://127.0.0.1:3000/api/storage/test.png"
        assert _is_local_storage_url(url) is True

    def test_127_0_0_1_without_port(self) -> None:
        """Test 127.0.0.1 URL without port."""
        url = "http://127.0.0.1/api/storage/test.png"
        assert _is_local_storage_url(url) is True

    def test_https_localhost(self) -> None:
        """Test HTTPS localhost URL."""
        url = "https://localhost:8443/api/storage/test.png"
        assert _is_local_storage_url(url) is True

    def test_non_local_url(self) -> None:
        """Test non-local URL."""
        url = "http://example.com/api/storage/test.png"
        assert _is_local_storage_url(url) is False

    def test_different_path(self) -> None:
        """Test URL with different path."""
        url = "http://localhost:7777/api/other/test.png"
        assert _is_local_storage_url(url) is False

    def test_http_with_default_port(self) -> None:
        """Test HTTP with default port 80."""
        url = "http://localhost:80/api/storage/test.png"
        assert _is_local_storage_url(url) is True


class TestExtractStorageKeyFromUrl:
    """Test _extract_storage_key_from_url function."""

    def test_extract_key_simple(self) -> None:
        """Test extracting simple key."""
        url = "http://localhost:7777/api/storage/test.png"
        key = _extract_storage_key_from_url(url)
        assert key == "test.png"

    def test_extract_key_with_uuid(self) -> None:
        """Test extracting UUID key."""
        url = "http://localhost:7777/api/storage/828ae5ded94411f0884a000022ae8b15.png"
        key = _extract_storage_key_from_url(url)
        assert key == "828ae5ded94411f0884a000022ae8b15.png"

    def test_extract_key_with_subdirs(self) -> None:
        """Test extracting key with subdirectories."""
        url = "http://localhost:7777/api/storage/path/to/file.jpg"
        key = _extract_storage_key_from_url(url)
        assert key == "path/to/file.jpg"

    def test_extract_key_no_extension(self) -> None:
        """Test extracting key without extension."""
        url = "http://localhost:7777/api/storage/file"
        key = _extract_storage_key_from_url(url)
        assert key == "file"

    def test_invalid_url_raises_error(self) -> None:
        """Test that invalid URL raises ValueError."""
        url = "http://localhost:7777/api/notstorage/test.png"
        with pytest.raises(ValueError, match="Could not extract storage key"):
            _extract_storage_key_from_url(url)


class TestParseAssetIdFromUri:
    """Test _parse_asset_id_from_uri function."""

    def test_parse_simple_asset_id(self) -> None:
        """Test parsing simple asset ID."""
        uri = "asset://abc123"
        asset_id = _parse_asset_id_from_uri(uri)
        assert asset_id == "abc123"

    def test_parse_asset_id_with_extension(self) -> None:
        """Test parsing asset ID with extension."""
        uri = "asset://abc123.png"
        asset_id = _parse_asset_id_from_uri(uri)
        assert asset_id == "abc123"

    def test_parse_asset_id_with_multiple_dots(self) -> None:
        """Test parsing asset ID with multiple dots."""
        uri = "asset://abc123.def.test.jpg"
        asset_id = _parse_asset_id_from_uri(uri)
        assert asset_id == "abc123"

    def test_parse_uuid_asset_id(self) -> None:
        """Test parsing UUID asset ID."""
        uri = "asset://828ae5ded94411f0884a000022ae8b15"
        asset_id = _parse_asset_id_from_uri(uri)
        assert asset_id == "828ae5ded94411f0884a000022ae8b15"

    def test_parse_asset_id_no_extension(self) -> None:
        """Test parsing without extension."""
        uri = "asset://my-asset-id"
        asset_id = _parse_asset_id_from_uri(uri)
        assert asset_id == "my-asset-id"

    def test_invalid_asset_uri_missing_prefix(self) -> None:
        """Test that URI without asset:// prefix raises error."""
        uri = "http://asset://abc123"
        with pytest.raises(ValueError, match="Invalid asset URI"):
            _parse_asset_id_from_uri(uri)

    def test_invalid_asset_uri_empty_id(self) -> None:
        """Test that URI with empty asset ID raises error."""
        uri = "asset://"
        with pytest.raises(ValueError, match="Invalid asset URI - no asset ID"):
            _parse_asset_id_from_uri(uri)

    def test_invalid_asset_uri_only_extension(self) -> None:
        """Test that URI with only extension raises error."""
        uri = "asset://.png"
        with pytest.raises(ValueError, match="Invalid asset URI - no asset ID"):
            _parse_asset_id_from_uri(uri)


class TestFetchUriBytesAndMimeAsync:
    """Test fetch_uri_bytes_and_mime_async function."""

    async def test_fetch_data_uri(self) -> None:
        """Test fetching data URI."""
        data = b"test data"
        b64_data = base64.b64encode(data).decode("utf-8")
        uri = f"data:text/plain;base64,{b64_data}"

        mime, result = await fetch_uri_bytes_and_mime_async(uri)

        assert mime == "text/plain"
        assert result == data

    @patch("nodetool.io.media_fetch.require_scope")
    async def test_fetch_memory_uri(self, mock_require_scope: Mock) -> None:
        """Test fetching memory URI."""
        # Create a test image
        img = PIL.Image.new("RGB", (10, 10), color="red")

        # Mock the scope and cache
        mock_scope = Mock()
        mock_cache = Mock()
        mock_cache.get.return_value = img
        mock_scope.get_memory_uri_cache.return_value = mock_cache
        mock_require_scope.return_value = mock_scope

        uri = "memory://test-obj"
        mime, result = await fetch_uri_bytes_and_mime_async(uri)

        assert mime == "image/png"
        assert isinstance(result, bytes)
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    @patch("builtins.open")
    async def test_fetch_file_uri(self, mock_open: Mock) -> None:
        """Test fetching file URI."""
        test_data = b"file content"
        mock_file = MagicMock()
        mock_file.read.return_value = test_data
        mock_open.return_value.__enter__.return_value = mock_file

        uri = "file:///tmp/test.txt"
        mime, result = await fetch_uri_bytes_and_mime_async(uri)

        assert result == test_data
        assert mime == "text/plain"

    async def test_unsupported_uri_scheme(self) -> None:
        """Test that unsupported URI scheme raises error."""
        uri = "ftp://example.com/file.txt"

        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            await fetch_uri_bytes_and_mime_async(uri)
