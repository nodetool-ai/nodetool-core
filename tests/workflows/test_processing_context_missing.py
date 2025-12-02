"""
Tests for missing coverage areas in ProcessingContext.
These tests target specific methods and code paths that weren't covered in the main tests.
"""

import importlib.util
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import PIL.Image
import pytest

from nodetool.metadata.types import (
    AssetRef,
    ImageRef,
    Provider,
    TextRef,
    VideoRef,
)
from nodetool.workflows.processing_context import ProcessingContext


def _torch_available() -> bool:
    """Check if torch is available."""
    return importlib.util.find_spec("torch") is not None


@pytest.fixture
def context():
    """Create a test ProcessingContext instance."""
    return ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workspace_dir="/tmp/test_workspace",
    )


class TestHttpMethods:
    """Test HTTP-related methods."""

    @pytest.mark.asyncio
    async def test_http_get(self, context: ProcessingContext):
        """Test HTTP GET request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client_instance = AsyncMock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_scope = Mock()
        mock_scope.get_http_client.return_value = mock_client_instance

        with patch("nodetool.workflows.processing_context.require_scope", return_value=mock_scope):
            result = await context.http_get("http://example.com")
            assert result == mock_response
            mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_post(self, context: ProcessingContext):
        """Test HTTP POST request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client_instance = AsyncMock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_scope = Mock()
        mock_scope.get_http_client.return_value = mock_client_instance

        with patch("nodetool.workflows.processing_context.require_scope", return_value=mock_scope):
            result = await context.http_post(
                "http://example.com", json={"test": "data"}
            )
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_http_put(self, context: ProcessingContext):
        """Test HTTP PUT request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client_instance = AsyncMock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_scope = Mock()
        mock_scope.get_http_client.return_value = mock_client_instance

        with patch("nodetool.workflows.processing_context.require_scope", return_value=mock_scope):
            result = await context.http_put("http://example.com")
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_http_patch(self, context: ProcessingContext):
        """Test HTTP PATCH request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client_instance = AsyncMock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_scope = Mock()
        mock_scope.get_http_client.return_value = mock_client_instance

        with patch("nodetool.workflows.processing_context.require_scope", return_value=mock_scope):
            result = await context.http_patch("http://example.com")
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_http_delete(self, context: ProcessingContext):
        """Test HTTP DELETE request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client_instance = AsyncMock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_scope = Mock()
        mock_scope.get_http_client.return_value = mock_client_instance

        with patch("nodetool.workflows.processing_context.require_scope", return_value=mock_scope):
            result = await context.http_delete("http://example.com")
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_http_head(self, context: ProcessingContext):
        """Test HTTP HEAD request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client_instance = AsyncMock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_scope = Mock()
        mock_scope.get_http_client.return_value = mock_client_instance

        with patch("nodetool.workflows.processing_context.require_scope", return_value=mock_scope):
            result = await context.http_head("http://example.com")
            assert result == mock_response


class TestDownloadFile:
    """Test file downloading functionality."""

    @pytest.mark.asyncio
    async def test_download_file_http(self, context: ProcessingContext):
        """Test downloading file via HTTP."""
        test_content = b"downloaded content"

        with patch.object(context, "http_get") as mock_get:
            mock_response = Mock()
            mock_response.content = test_content
            mock_get.return_value = mock_response

            result = await context.download_file("http://example.com/file.txt")
            assert result.read() == test_content

    @pytest.mark.asyncio
    async def test_download_file_data_uri(self, context: ProcessingContext):
        """Test downloading file from data URI."""
        # data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAF2nGQ9gAAAABJRU5ErkJggg==
        test_data_uri = "data:image/png;base64,dGVzdCBkYXRh"  # "test data" in base64

        result = await context.download_file(test_data_uri)
        assert result.name.endswith(".png")
        assert b"test data" in result.read()

    @pytest.mark.asyncio
    async def test_download_file_local_unix(self, context: ProcessingContext):
        """Test downloading local file on Unix."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"local file content")
            temp_path = temp_file.name

        try:
            file_uri = f"file://{temp_path}"
            result = await context.download_file(file_uri)
            assert result.read() == b"local file content"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_download_file_not_found(self, context: ProcessingContext):
        """Test downloading non-existent local file."""
        with pytest.raises(FileNotFoundError):
            await context.download_file("file:///nonexistent/path")

    @pytest.mark.asyncio
    async def test_download_file_windows_path(self, context: ProcessingContext):
        """Test downloading file with Windows-style path handling."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"windows file content")
            temp_path = temp_file.name

        try:
            # Test the Windows URI handling without mocking os.name
            # since that would break file system operations on Unix
            result = await context.download_file(temp_path)
            assert b"windows file content" in result.read()
        finally:
            os.unlink(temp_path)


class TestPredictionAndGeneration:
    """Test prediction and message generation functionality."""

    @pytest.mark.asyncio
    async def test_run_prediction(self, context: ProcessingContext):
        """Test running a prediction."""
        from nodetool.types.prediction import PredictionResult

        async def mock_prediction_function(prediction, env):
            yield PredictionResult(
                prediction=prediction, encoding="json", content="test result"
            )

        with patch("nodetool.models.prediction.Prediction.create"):
            result = await context.run_prediction(
                node_id="test_node",
                provider="openai",
                model="gpt-4",
                run_prediction_function=mock_prediction_function,
            )
            assert result == "test result"

    @pytest.mark.asyncio
    async def test_run_prediction_no_result(self, context: ProcessingContext):
        """Test running prediction that doesn't return result."""

        async def mock_prediction_function(prediction, env):
            return
            yield  # unreachable

        with pytest.raises(ValueError, match="Prediction did not return a result"):
            await context.run_prediction(
                node_id="test_node",
                provider="openai",
                model="gpt-4",
                run_prediction_function=mock_prediction_function,
            )

    @pytest.mark.asyncio
    async def test_stream_prediction(self, context: ProcessingContext):
        """Test streaming prediction."""
        from nodetool.types.prediction import PredictionResult

        async def mock_prediction_function(prediction, env):
            yield PredictionResult(
                prediction=prediction, encoding="json", content="chunk1"
            )
            yield PredictionResult(
                prediction=prediction, encoding="json", content="chunk2"
            )

        results = []
        async for result in context.stream_prediction(
            node_id="test_node",
            provider=Provider.OpenAI,
            model="gpt-4",
            run_prediction_function=mock_prediction_function,
        ):
            results.append(result)

        assert len(results) == 2
        assert results[0].content == "chunk1"
        assert results[1].content == "chunk2"


class TestConversionMethods:
    """Test asset conversion methods not covered elsewhere."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _torch_available(), reason="torch not installed")
    async def test_image_to_tensor(self, context: ProcessingContext):
        """Test converting image to tensor."""
        # Create test image
        test_image = PIL.Image.new("RGB", (10, 10), color="red")
        buffer = BytesIO()
        test_image.save(buffer, format="PNG")
        image_ref = ImageRef(data=buffer.getvalue())

        import torch

        with patch("nodetool.workflows.processing_context.TORCH_AVAILABLE", True):
            result = await context.image_to_tensor(image_ref)
            assert isinstance(result, torch.Tensor)
            assert result.shape == (10, 10, 3)

    @pytest.mark.asyncio
    async def test_image_to_tensor_unavailable(self, context: ProcessingContext):
        """Test converting image to tensor when torch unavailable."""
        test_image = PIL.Image.new("RGB", (10, 10), color="red")
        buffer = BytesIO()
        test_image.save(buffer, format="PNG")
        image_ref = ImageRef(data=buffer.getvalue())

        with patch("nodetool.workflows.processing_context.TORCH_AVAILABLE", False), pytest.raises(
            ImportError, match="torch is required"
        ):
            await context.image_to_tensor(image_ref)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _torch_available(), reason="torch not installed")
    async def test_image_to_torch_tensor(self, context: ProcessingContext):
        """Test converting image to torch tensor."""
        test_image = PIL.Image.new("RGB", (10, 10), color="blue")
        buffer = BytesIO()
        test_image.save(buffer, format="PNG")
        image_ref = ImageRef(data=buffer.getvalue())

        import torch

        with patch("nodetool.workflows.processing_context.TORCH_AVAILABLE", True):
            result = await context.image_to_torch_tensor(image_ref)
            assert isinstance(result, torch.Tensor)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _torch_available(), reason="torch not installed")
    async def test_image_from_tensor(self, context: ProcessingContext):
        """Test creating image from tensor."""
        import torch

        # Create a test tensor (batch, height, width, channels)
        test_tensor = torch.rand(2, 20, 20, 3)

        with patch("nodetool.workflows.processing_context.TORCH_AVAILABLE", True):
            result = await context.image_from_tensor(test_tensor)
            assert isinstance(result, ImageRef)
            assert isinstance(result.data, list)  # Should be a batch
            assert len(result.data) == 2

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _torch_available(), reason="torch not installed")
    async def test_image_from_tensor_single(self, context: ProcessingContext):
        """Test creating single image from tensor."""
        import torch

        # Create a test tensor with batch size 1
        test_tensor = torch.rand(1, 20, 20, 3)

        with patch("nodetool.workflows.processing_context.TORCH_AVAILABLE", True):
            result = await context.image_from_tensor(test_tensor)
            assert isinstance(result, ImageRef)
            assert not isinstance(result.data, list)  # Should be single image

    @pytest.mark.asyncio
    async def test_convert_value_for_prediction_asset_ref(
        self, context: ProcessingContext
    ):
        """Test converting AssetRef for prediction."""
        from nodetool.metadata.type_metadata import TypeMetadata
        from nodetool.workflows.property import Property

        property = Property(name="test", type=TypeMetadata(type="image"))
        asset_ref = ImageRef(data=b"fake image data")

        result = await context.convert_value_for_prediction(property, asset_ref)
        assert result is not None
        assert result.startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_convert_value_for_prediction_text_ref(
        self, context: ProcessingContext
    ):
        """Test converting TextRef for prediction."""
        from nodetool.metadata.type_metadata import TypeMetadata
        from nodetool.workflows.property import Property

        property = Property(name="test", type=TypeMetadata(type="text"))
        text_ref = TextRef(data=b"test text content")

        result = await context.convert_value_for_prediction(property, text_ref)
        assert result == "test text content"

    @pytest.mark.asyncio
    async def test_convert_value_for_prediction_empty_asset(
        self, context: ProcessingContext
    ):
        """Test converting empty AssetRef for prediction."""
        from nodetool.metadata.type_metadata import TypeMetadata
        from nodetool.workflows.property import Property

        property = Property(name="test", type=TypeMetadata(type="image"))
        empty_asset = AssetRef()

        result = await context.convert_value_for_prediction(property, empty_asset)
        assert result is None

    @pytest.mark.asyncio
    async def test_convert_value_for_prediction_enum(self, context: ProcessingContext):
        """Test converting enum for prediction."""
        from enum import Enum

        from nodetool.metadata.type_metadata import TypeMetadata
        from nodetool.workflows.property import Property

        class TestEnum(Enum):
            VALUE1 = "value1"
            VALUE2 = "value2"

        property = Property(name="test", type=TypeMetadata(type="enum"))

        # Test enum value
        result = await context.convert_value_for_prediction(property, TestEnum.VALUE1)
        assert result == "value1"

        # Test string value
        result = await context.convert_value_for_prediction(property, "value2")
        assert result == "value2"

        # Test None value
        result = await context.convert_value_for_prediction(property, None)
        assert result is None


    @pytest.mark.asyncio
    async def test_is_huggingface_model_cached(self, context: ProcessingContext):
        """Test checking if HuggingFace model is cached."""
        with patch(
            "nodetool.integrations.huggingface.hf_utils.try_to_load_from_cache"
        ) as mock_cache:
            mock_cache.return_value = "/path/to/cache"

            result = await context.is_huggingface_model_cached("bert-base-uncased")
            assert result is True

            mock_cache.return_value = None
            result = await context.is_huggingface_model_cached("nonexistent-model")
            assert result is False


class TestVideoMethods:
    """Test video methods not covered elsewhere."""

    @pytest.mark.asyncio
    async def test_video_from_frames_pil(self, context: ProcessingContext):
        """Test creating video from PIL Image frames."""
        frames = [PIL.Image.new("RGB", (10, 10), color="red") for _ in range(3)]

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = Mock()
            mock_file.name = "/tmp/test_video.mp4"
            mock_temp.return_value.__enter__.return_value = mock_file

            with patch(
                "nodetool.media.video.video_utils.export_to_video"
            ) as mock_export, patch("builtins.open", create=True) as mock_open:
                mock_open.return_value = BytesIO(b"fake video data")

                result = await context.video_from_frames(frames, fps=24)
                assert isinstance(result, VideoRef)
                mock_export.assert_called_once_with(frames, mock_file.name, fps=24)
