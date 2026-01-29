"""Tests for the asset_storage module."""

import base64
import pytest
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    DataframeRef,
    ImageRef,
    JSONRef,
    SVGRef,
    TextRef,
    VideoRef,
)
from nodetool.workflows.asset_storage import (
    find_asset_refs,
    get_content_type_for_asset_ref,
    object_to_bytes,
    convert_asset_data_to_bytes,
    decode_data_uri,
    read_file_uri,
    download_http_uri,
    resolve_asset_content,
    auto_save_assets,
)


class TestFindAssetRefs:
    """Tests for find_asset_refs function."""

    def test_find_single_asset_ref(self):
        """Test finding a single AssetRef in a dict."""
        asset = ImageRef(uri="test://image")
        result = {"output": asset}

        refs = find_asset_refs(result)

        assert len(refs) == 1
        assert refs[0] == ("output", asset)

    def test_find_nested_asset_ref(self):
        """Test finding AssetRef in nested dict."""
        asset = ImageRef(uri="test://image")
        result = {"data": {"nested": {"image": asset}}}

        refs = find_asset_refs(result)

        assert len(refs) == 1
        assert refs[0] == ("data.nested.image", asset)

    def test_find_asset_ref_in_list(self):
        """Test finding AssetRef in a list."""
        asset1 = ImageRef(uri="test://image1")
        asset2 = ImageRef(uri="test://image2")
        result = {"images": [asset1, asset2]}

        refs = find_asset_refs(result)

        assert len(refs) == 2
        assert refs[0] == ("images[0]", asset1)
        assert refs[1] == ("images[1]", asset2)

    def test_find_no_asset_refs(self):
        """Test with no AssetRefs present."""
        result = {"value": 42, "text": "hello"}

        refs = find_asset_refs(result)

        assert len(refs) == 0

    def test_find_mixed_content(self):
        """Test with mixed content types."""
        asset = TextRef(uri="test://text")
        result = {
            "number": 123,
            "nested": {
                "items": [1, 2, {"asset": asset}],
                "other": "string",
            },
        }

        refs = find_asset_refs(result)

        assert len(refs) == 1
        assert refs[0] == ("nested.items[2].asset", asset)


class TestGetContentTypeForAssetRef:
    """Tests for get_content_type_for_asset_ref function."""

    @pytest.mark.parametrize(
        "asset_ref,expected_type",
        [
            (ImageRef(uri="test://"), "image/png"),
            (AudioRef(uri="test://"), "audio/mp3"),
            (VideoRef(uri="test://"), "video/mp4"),
            (TextRef(uri="test://"), "text/plain"),
            (DataframeRef(uri="test://"), "application/json"),
            (JSONRef(uri="test://"), "application/json"),
            (SVGRef(uri="test://"), "image/svg+xml"),
        ],
    )
    def test_content_types(self, asset_ref, expected_type):
        """Test content type mapping for various asset types."""
        assert get_content_type_for_asset_ref(asset_ref) == expected_type

    def test_unknown_type_returns_octet_stream(self):
        """Test that unknown types return application/octet-stream."""
        asset = AssetRef(uri="test://")
        assert get_content_type_for_asset_ref(asset) == "application/octet-stream"


class TestConvertAssetDataToBytes:
    """Tests for convert_asset_data_to_bytes function."""

    def test_convert_dataframe_ref(self):
        """Test converting DataframeRef data to bytes."""
        # DataframeRef expects data as list of lists (rows)
        data = [[1, 2], [3, 4]]
        asset = DataframeRef(uri="test://", data=data)

        result = convert_asset_data_to_bytes(asset, "test.path")

        assert result is not None
        assert b"[[1, 2], [3, 4]]" in result.getvalue()

    def test_convert_json_ref_string(self):
        """Test converting JSONRef with string data."""
        asset = JSONRef(uri="test://", data='{"key": "value"}')

        result = convert_asset_data_to_bytes(asset, "test.path")

        assert result is not None
        assert result.getvalue() == b'{"key": "value"}'

    def test_convert_svg_ref_string(self):
        """Test converting SVGRef with string data."""
        svg_data = '<svg><circle cx="50" cy="50" r="40"/></svg>'
        asset = SVGRef(uri="test://", data=svg_data)  # type: ignore[arg-type]

        result = convert_asset_data_to_bytes(asset, "test.path")

        assert result is not None
        assert result.getvalue() == svg_data.encode("utf-8")

    def test_convert_bytes_data(self):
        """Test converting bytes data directly."""
        data = b"binary data"
        asset = ImageRef(uri="test://", data=data)

        result = convert_asset_data_to_bytes(asset, "test.path")

        assert result is not None
        assert result.getvalue() == data


class TestDecodeDataUri:
    """Tests for decode_data_uri function."""

    def test_decode_base64_data_uri(self):
        """Test decoding a base64 data URI."""
        data = b"Hello, World!"
        encoded = base64.b64encode(data).decode("ascii")
        uri = f"data:text/plain;base64,{encoded}"

        result = decode_data_uri(uri, "test.path")

        assert result is not None
        assert result.getvalue() == data

    def test_decode_url_encoded_data_uri(self):
        """Test decoding a URL-encoded data URI."""
        uri = "data:text/plain,Hello%20World"

        result = decode_data_uri(uri, "test.path")

        assert result is not None
        assert result.getvalue() == b"Hello World"

    def test_decode_invalid_base64(self):
        """Test handling invalid base64 data."""
        uri = "data:text/plain;base64,not-valid-base64!!!"

        result = decode_data_uri(uri, "test.path")

        assert result is None

    def test_decode_malformed_uri(self):
        """Test handling malformed data URI (no comma)."""
        uri = "data:text/plain;base64"

        result = decode_data_uri(uri, "test.path")

        assert result is None


class TestReadFileUri:
    """Tests for read_file_uri function."""

    def test_read_file_uri(self, tmp_path):
        """Test reading a file:// URI."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"file content")
        uri = f"file://{test_file}"

        result = read_file_uri(uri, "test.path")

        assert result is not None
        assert result.getvalue() == b"file content"

    def test_read_nonexistent_file(self):
        """Test reading a non-existent file."""
        uri = "file:///nonexistent/path/to/file.txt"

        result = read_file_uri(uri, "test.path")

        assert result is None

    def test_read_file_uri_with_url_encoding(self, tmp_path):
        """Test reading a file with URL-encoded path."""
        test_file = tmp_path / "test file.txt"
        test_file.write_bytes(b"content with spaces")
        uri = f"file://{str(test_file).replace(' ', '%20')}"

        result = read_file_uri(uri, "test.path")

        assert result is not None
        assert result.getvalue() == b"content with spaces"


class TestDownloadHttpUri:
    """Tests for download_http_uri function."""

    @pytest.mark.asyncio
    async def test_download_http_success(self):
        """Test successful HTTP download."""
        import httpx

        mock_response = MagicMock()
        mock_response.content = b"downloaded content"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(httpx, "AsyncClient") as mock_async_client:
            mock_async_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await download_http_uri("http://example.com/file.png", "test.path")

            assert result is not None
            assert result.getvalue() == b"downloaded content"

    @pytest.mark.asyncio
    async def test_download_http_failure(self):
        """Test HTTP download failure."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404)
        ))

        with patch.object(httpx, "AsyncClient") as mock_async_client:
            mock_async_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await download_http_uri("http://example.com/file.png", "test.path")

            assert result is None


class TestResolveAssetContent:
    """Tests for resolve_asset_content function."""

    @pytest.mark.asyncio
    async def test_resolve_with_data(self):
        """Test resolving asset with direct data."""
        asset = ImageRef(uri="test://", data=b"image bytes")

        result = await resolve_asset_content(asset, "test.path")

        assert result is not None
        assert result.getvalue() == b"image bytes"

    @pytest.mark.asyncio
    async def test_resolve_data_uri(self):
        """Test resolving asset with data: URI."""
        data = b"test data"
        encoded = base64.b64encode(data).decode("ascii")
        asset = ImageRef(uri=f"data:image/png;base64,{encoded}")

        result = await resolve_asset_content(asset, "test.path")

        assert result is not None
        assert result.getvalue() == data

    @pytest.mark.asyncio
    async def test_resolve_file_uri(self, tmp_path):
        """Test resolving asset with file: URI."""
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"file content")
        asset = ImageRef(uri=f"file://{test_file}")

        result = await resolve_asset_content(asset, "test.path")

        assert result is not None
        assert result.getvalue() == b"file content"

    @pytest.mark.asyncio
    async def test_resolve_unsupported_uri(self):
        """Test resolving asset with unsupported URI scheme."""
        asset = ImageRef(uri="ftp://example.com/file.png")

        result = await resolve_asset_content(asset, "test.path")

        assert result is None


class TestAutoSaveAssets:
    """Tests for auto_save_assets function."""

    @pytest.mark.asyncio
    async def test_auto_save_with_data(self):
        """Test auto-saving an asset with direct data."""
        node = MagicMock()
        node.get_title.return_value = "TestNode"
        node._id = "abc12345"

        asset = ImageRef(uri="test://", data=b"image bytes")
        result = {"output": asset}

        context = MagicMock()
        mock_asset = MagicMock()
        mock_asset.id = "saved-asset-id"
        context.create_asset = AsyncMock(return_value=mock_asset)

        await auto_save_assets(node, result, context)

        context.create_asset.assert_called_once()
        assert asset.asset_id == "saved-asset-id"
        assert asset.uri == "asset://saved-asset-id.png"

    @pytest.mark.asyncio
    async def test_skip_already_saved_asset(self):
        """Test that assets with asset_id are skipped."""
        node = MagicMock()
        node.get_title.return_value = "TestNode"
        node._id = "abc12345"

        asset = ImageRef(uri="test://", data=b"image bytes", asset_id="existing-id")
        result = {"output": asset}

        context = MagicMock()
        context.create_asset = AsyncMock()

        await auto_save_assets(node, result, context)

        context.create_asset.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_result(self):
        """Test handling empty result."""
        node = MagicMock()
        context = MagicMock()
        context.create_asset = AsyncMock()

        await auto_save_assets(node, {}, context)

        context.create_asset.assert_not_called()


class TestObjectToBytes:
    """Tests for object_to_bytes function."""

    def test_convert_string_to_bytes(self):
        """Test converting string to bytes for TextRef."""
        asset = TextRef(uri="test://")

        result = object_to_bytes("hello world", asset)

        assert result == b"hello world"

    def test_convert_bytes_passthrough(self):
        """Test that bytes are passed through unchanged."""
        asset = ImageRef(uri="test://")

        result = object_to_bytes(b"binary data", asset)

        assert result == b"binary data"

    def test_convert_pil_image(self):
        """Test converting PIL Image to bytes."""
        from PIL import Image

        asset = ImageRef(uri="test://")
        img = Image.new("RGB", (10, 10), color="red")

        result = object_to_bytes(img, asset)

        assert result is not None
        assert len(result) > 0
        # Verify it's a valid PNG
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_unsupported_object_type(self):
        """Test handling unsupported object type."""
        asset = ImageRef(uri="test://")

        result = object_to_bytes({"not": "an image"}, asset)

        assert result is None
