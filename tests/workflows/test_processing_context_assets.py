"""
Tests for ProcessingContext asset conversion and manipulation methods.
Covers all the asset creation, conversion, and utility methods.
"""

import pytest
import asyncio
from io import BytesIO
from unittest.mock import Mock, patch, AsyncMock
import base64
import numpy as np
import PIL.Image
import pandas as pd
from pydub import AudioSegment

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import (
    AssetRef, ImageRef, AudioRef, VideoRef, TextRef, 
    DataframeRef, ModelRef
)
from nodetool.common.environment import Environment


@pytest.fixture
def context():
    """Create a test ProcessingContext instance."""
    return ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workspace_dir="/tmp/test_workspace"
    )


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return PIL.Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def sample_audio_segment():
    """Create a sample AudioSegment for testing."""
    # Create a simple sine wave
    import math
    sample_rate = 44100
    duration = 1.0  # 1 second
    frequency = 440  # A4 note
    
    samples = []
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        sample = int(32767 * math.sin(2 * math.pi * frequency * t))
        samples.append(sample)
    
    audio_data = bytearray()
    for sample in samples:
        audio_data.extend(sample.to_bytes(2, byteorder='little', signed=True))
    
    return AudioSegment(
        data=bytes(audio_data),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1
    )


class TestAssetConversion:
    """Test asset conversion methods."""
    
    @pytest.mark.asyncio
    async def test_asset_to_io(self, context):
        """Test converting AssetRef to IO object."""
        # Test with data
        test_data = b"test content"
        asset_ref = AssetRef(data=test_data)
        
        io_obj = await context.asset_to_io(asset_ref)
        assert io_obj.read() == test_data
    
    @pytest.mark.asyncio
    async def test_asset_to_io_with_asset_id(self, context):
        """Test converting AssetRef with asset_id to IO object."""
        asset_ref = AssetRef(asset_id="test_asset_id")
        
        with patch.object(context, 'download_asset', return_value=BytesIO(b"downloaded content")) as mock_download:
            io_obj = await context.asset_to_io(asset_ref)
            assert io_obj.read() == b"downloaded content"
            mock_download.assert_called_once_with("test_asset_id")
    
    @pytest.mark.asyncio
    async def test_asset_to_io_with_uri(self, context):
        """Test converting AssetRef with URI to IO object."""
        asset_ref = AssetRef(uri="http://example.com/file.txt")
        
        with patch.object(context, 'download_file', return_value=BytesIO(b"uri content")) as mock_download:
            io_obj = await context.asset_to_io(asset_ref)
            assert io_obj.read() == b"uri content"
            mock_download.assert_called_once_with("http://example.com/file.txt")
    
    @pytest.mark.asyncio
    async def test_asset_to_io_empty_asset(self, context):
        """Test converting empty AssetRef raises error."""
        asset_ref = AssetRef()
        
        with pytest.raises(ValueError, match="AssetRef is empty"):
            await context.asset_to_io(asset_ref)
    
    @pytest.mark.asyncio
    async def test_asset_to_io_batched_data_error(self, context):
        """Test converting AssetRef with batched data raises error."""
        asset_ref = AssetRef(data=[b"data1", b"data2"])
        
        with pytest.raises(ValueError, match="Batched data must be converted"):
            await context.asset_to_io(asset_ref)
    
    @pytest.mark.asyncio
    async def test_asset_to_bytes(self, context):
        """Test converting AssetRef to bytes."""
        test_data = b"test content"
        asset_ref = AssetRef(data=test_data)
        
        result = await context.asset_to_bytes(asset_ref)
        assert result == test_data


class TestImageMethods:
    """Test image-related methods."""
    
    @pytest.mark.asyncio
    async def test_image_to_pil(self, context, sample_image):
        """Test converting ImageRef to PIL Image."""
        # Create ImageRef with image data
        buffer = BytesIO()
        sample_image.save(buffer, format='PNG')
        image_ref = ImageRef(data=buffer.getvalue())
        
        result = await context.image_to_pil(image_ref)
        assert isinstance(result, PIL.Image.Image)
        assert result.size == (100, 100)
        assert result.mode == 'RGB'
    
    @pytest.mark.asyncio
    async def test_image_to_numpy(self, context, sample_image):
        """Test converting ImageRef to numpy array."""
        buffer = BytesIO()
        sample_image.save(buffer, format='PNG')
        image_ref = ImageRef(data=buffer.getvalue())
        
        result = await context.image_to_numpy(image_ref)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)  # height, width, channels
    
    @pytest.mark.asyncio
    async def test_image_to_base64(self, context, sample_image):
        """Test converting ImageRef to base64 string."""
        buffer = BytesIO()
        sample_image.save(buffer, format='PNG')
        image_ref = ImageRef(data=buffer.getvalue())
        
        result = await context.image_to_base64(image_ref)
        assert isinstance(result, str)
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
    
    @pytest.mark.asyncio
    async def test_image_from_io(self, context):
        """Test creating ImageRef from IO object."""
        # Create test image data
        test_image = PIL.Image.new('RGB', (50, 50), color='blue')
        buffer = BytesIO()
        test_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Test without name (should create data-only ImageRef)
        result = await context.image_from_io(buffer)
        assert isinstance(result, ImageRef)
        assert result.data is not None
        assert result.asset_id is None
    
    @pytest.mark.asyncio
    async def test_image_from_io_with_name(self, context):
        """Test creating ImageRef from IO object with name."""
        test_image = PIL.Image.new('RGB', (50, 50), color='blue')
        buffer = BytesIO()
        test_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        with patch.object(context, 'create_asset') as mock_create:
            mock_asset = Mock()
            mock_asset.id = "test_asset_id"
            mock_asset.get_url = "http://test.com/image.png"
            mock_create.return_value = mock_asset
            
            with patch.object(Environment, 'get_asset_storage') as mock_storage:
                mock_storage_instance = Mock()
                mock_storage_instance.get_url.return_value = "http://test.com/image.png"
                mock_storage.return_value = mock_storage_instance
                
                result = await context.image_from_io(buffer, name="test.png")
                assert isinstance(result, ImageRef)
                assert result.asset_id == "test_asset_id"
                assert result.uri == "http://test.com/image.png"
    
    @pytest.mark.asyncio
    async def test_image_from_bytes(self, context):
        """Test creating ImageRef from bytes."""
        test_image = PIL.Image.new('RGB', (50, 50), color='green')
        buffer = BytesIO()
        test_image.save(buffer, format='PNG')
        test_bytes = buffer.getvalue()
        
        result = await context.image_from_bytes(test_bytes)
        assert isinstance(result, ImageRef)
        assert result.data == test_bytes
    
    @pytest.mark.asyncio
    async def test_image_from_base64(self, context):
        """Test creating ImageRef from base64 string."""
        test_image = PIL.Image.new('RGB', (50, 50), color='yellow')
        buffer = BytesIO()
        test_image.save(buffer, format='PNG')
        test_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        result = await context.image_from_base64(test_b64)
        assert isinstance(result, ImageRef)
        assert result.data is not None
    
    @pytest.mark.asyncio
    async def test_image_from_pil(self, context, sample_image):
        """Test creating ImageRef from PIL Image."""
        result = await context.image_from_pil(sample_image)
        assert isinstance(result, ImageRef)
        assert result.uri.startswith("memory://")  # Should be a memory URI
        
        # Verify we can convert back to PIL
        recovered_image = await context.image_to_pil(result)
        assert recovered_image.size == sample_image.size
    
    @pytest.mark.asyncio
    async def test_image_from_numpy(self, context):
        """Test creating ImageRef from numpy array."""
        # Create test numpy array
        test_array = np.full((50, 50, 3), fill_value=128, dtype=np.uint8)
        
        result = await context.image_from_numpy(test_array)
        assert isinstance(result, ImageRef)
        assert result.uri.startswith("memory://")  # Should be a memory URI
    
    @pytest.mark.asyncio
    async def test_image_from_url(self, context):
        """Test creating ImageRef from URL."""
        test_image = PIL.Image.new('RGB', (30, 30), color='purple')
        buffer = BytesIO()
        test_image.save(buffer, format='PNG')
        
        with patch.object(context, 'download_file', return_value=BytesIO(buffer.getvalue())):
            result = await context.image_from_url("http://example.com/image.png")
            assert isinstance(result, ImageRef)
            assert result.data is not None


class TestAudioMethods:
    """Test audio-related methods."""
    
    @pytest.mark.asyncio
    async def test_audio_to_audio_segment(self, context):
        """Test converting AudioRef to AudioSegment."""
        # Create simple audio data
        audio_data = b'\x00\x00' * 1000  # Simple silent audio
        audio_ref = AudioRef(data=audio_data)
        
        with patch('pydub.AudioSegment.from_file') as mock_from_file:
            mock_segment = Mock()
            mock_from_file.return_value = mock_segment
            
            result = await context.audio_to_audio_segment(audio_ref)
            assert result == mock_segment
    
    @pytest.mark.asyncio
    async def test_audio_to_numpy(self, context, sample_audio_segment):
        """Test converting AudioRef to numpy array."""
        # Create AudioRef from segment
        buffer = BytesIO()
        sample_audio_segment.export(buffer, format="wav")
        audio_ref = AudioRef(data=buffer.getvalue())
        
        with patch.object(context, 'audio_to_audio_segment', return_value=sample_audio_segment):
            samples, frame_rate, channels = await context.audio_to_numpy(audio_ref)
            assert isinstance(samples, np.ndarray)
            assert samples.dtype == np.float32
            assert frame_rate == 32000  # Default sample rate
            assert channels == 1
    
    @pytest.mark.asyncio
    async def test_audio_to_base64(self, context):
        """Test converting AudioRef to base64 string."""
        test_data = b"fake audio data"
        audio_ref = AudioRef(data=test_data)
        
        result = await context.audio_to_base64(audio_ref)
        assert isinstance(result, str)
        assert base64.b64decode(result) == test_data
    
    @pytest.mark.asyncio
    async def test_audio_from_io(self, context):
        """Test creating AudioRef from IO object."""
        buffer = BytesIO(b"audio content")
        
        result = await context.audio_from_io(buffer)
        assert isinstance(result, AudioRef)
        assert result.data == b"audio content"
    
    @pytest.mark.asyncio
    async def test_audio_from_bytes(self, context):
        """Test creating AudioRef from bytes."""
        test_bytes = b"audio bytes"
        
        result = await context.audio_from_bytes(test_bytes)
        assert isinstance(result, AudioRef)
        assert result.data == test_bytes
    
    @pytest.mark.asyncio
    async def test_audio_from_base64(self, context):
        """Test creating AudioRef from base64 string."""
        test_data = b"audio data"
        test_b64 = base64.b64encode(test_data).decode('utf-8')
        
        result = await context.audio_from_base64(test_b64)
        assert isinstance(result, AudioRef)
        assert result.data == test_data
    
    @pytest.mark.asyncio
    async def test_audio_from_numpy(self, context):
        """Test creating AudioRef from numpy array."""
        # Create test audio data as int16
        test_data = np.array([100, -100, 200, -200], dtype=np.int16)
        
        result = await context.audio_from_numpy(test_data, sample_rate=22050)
        assert isinstance(result, AudioRef)
        assert result.uri.startswith("memory://")  # Should be a memory URI
    
    @pytest.mark.asyncio
    async def test_audio_from_segment(self, context, sample_audio_segment):
        """Test creating AudioRef from AudioSegment."""
        result = await context.audio_from_segment(sample_audio_segment)
        assert isinstance(result, AudioRef)
        assert result.uri.startswith("memory://")  # Should be a memory URI


class TestTextMethods:
    """Test text-related methods."""
    
    @pytest.mark.asyncio
    async def test_text_to_str_with_textref(self, context):
        """Test converting TextRef to string."""
        test_text = "Hello, world!"
        text_ref = TextRef(data=test_text.encode('utf-8'))
        
        result = await context.text_to_str(text_ref)
        assert result == test_text
    
    @pytest.mark.asyncio
    async def test_text_to_str_with_string(self, context):
        """Test passing string directly to text_to_str."""
        test_text = "Direct string"
        
        result = await context.text_to_str(test_text)
        assert result == test_text
    
    @pytest.mark.asyncio
    async def test_text_from_str(self, context):
        """Test creating TextRef from string."""
        test_text = "Test string content"
        
        result = await context.text_from_str(test_text)
        assert isinstance(result, TextRef)
        assert result.uri.startswith("memory://")  # Should be a memory URI
    
    @pytest.mark.asyncio
    async def test_text_from_str_with_name(self, context):
        """Test creating TextRef from string with name."""
        test_text = "Named text content"
        
        with patch.object(context, 'create_asset') as mock_create:
            mock_asset = Mock()
            mock_asset.id = "text_asset_id"
            mock_asset.get_url = "http://test.com/text.txt"
            mock_create.return_value = mock_asset
            
            with patch.object(Environment, 'get_asset_storage') as mock_storage:
                mock_storage_instance = Mock()
                mock_storage_instance.get_url.return_value = "http://test.com/text.txt"
                mock_storage.return_value = mock_storage_instance
                
                result = await context.text_from_str(test_text, name="test.txt")
                assert isinstance(result, TextRef)
                assert result.asset_id == "text_asset_id"


class TestDataFrameMethods:
    """Test DataFrame-related methods."""
    
    @pytest.mark.asyncio
    async def test_dataframe_to_pandas_with_columns(self, context):
        """Test converting DataframeRef with columns to pandas DataFrame."""
        from nodetool.metadata.types import ColumnDef
        
        columns = [
            ColumnDef(name="col1", data_type="int"),
            ColumnDef(name="col2", data_type="string")
        ]
        data = [[1, "a"], [2, "b"], [3, "c"]]
        
        df_ref = DataframeRef(columns=columns, data=data)
        
        result = await context.dataframe_to_pandas(df_ref)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]
        assert len(result) == 3
    
    @pytest.mark.asyncio
    async def test_dataframe_to_pandas_memory_uri(self, context):
        """Test converting DataframeRef with memory URI."""
        test_df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        memory_uri = "memory://test_df"
        context.memory[memory_uri] = test_df
        
        df_ref = DataframeRef(uri=memory_uri)
        
        result = await context.dataframe_to_pandas(df_ref)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, test_df)
    
    @pytest.mark.asyncio
    async def test_dataframe_from_pandas(self, context):
        """Test creating DataframeRef from pandas DataFrame."""
        test_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        
        result = await context.dataframe_from_pandas(test_df)
        assert isinstance(result, DataframeRef)
        assert result.uri.startswith("memory://")
        
        # Verify we can convert back
        recovered_df = await context.dataframe_to_pandas(result)
        pd.testing.assert_frame_equal(recovered_df, test_df)


class TestVideoMethods:
    """Test video-related methods."""
    
    @pytest.mark.asyncio
    async def test_video_from_io(self, context):
        """Test creating VideoRef from IO object."""
        buffer = BytesIO(b"fake video data")
        
        result = await context.video_from_io(buffer)
        assert isinstance(result, VideoRef)
        assert result.data == b"fake video data"
    
    @pytest.mark.asyncio
    async def test_video_from_bytes(self, context):
        """Test creating VideoRef from bytes."""
        test_bytes = b"video bytes"
        
        result = await context.video_from_bytes(test_bytes)
        assert isinstance(result, VideoRef)
        assert result.data == test_bytes
    
    @pytest.mark.asyncio
    async def test_video_from_numpy(self, context):
        """Test creating VideoRef from numpy array."""
        # Create fake video data (frames, height, width, channels)
        video_array = np.random.randint(0, 255, (10, 50, 50, 3), dtype=np.uint8)
        
        with patch('imageio.mimwrite') as mock_mimwrite:
            def mock_write(buffer, video_data, format=None, fps=None):
                buffer.write(b"fake video output")
            
            mock_mimwrite.side_effect = mock_write
            
            result = await context.video_from_numpy(video_array, fps=30)
            assert isinstance(result, VideoRef)
            assert result.data is not None


class TestModelMethods:
    """Test ML model-related methods."""
    
    @pytest.mark.asyncio
    async def test_to_estimator(self, context):
        """Test converting ModelRef to estimator."""
        # Create fake model data
        import joblib
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        buffer = BytesIO()
        joblib.dump(model, buffer)
        
        model_ref = ModelRef(asset_id="test_model_id", data=buffer.getvalue())
        
        with patch('joblib.load') as mock_load:
            mock_load.return_value = model
            
            result = await context.to_estimator(model_ref)
            assert result == model
    
    @pytest.mark.asyncio
    async def test_to_estimator_no_asset_id(self, context):
        """Test converting empty ModelRef raises error."""
        model_ref = ModelRef()
        
        with pytest.raises(ValueError, match="ModelRef is empty"):
            await context.to_estimator(model_ref)
    
    @pytest.mark.asyncio
    async def test_from_estimator(self, context):
        """Test creating ModelRef from estimator."""
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        
        with patch.object(context, 'create_asset') as mock_create:
            mock_asset = Mock()
            mock_asset.id = "model_asset_id"
            mock_asset.file_name = "model.joblib"
            mock_create.return_value = mock_asset
            
            with patch.object(Environment, 'get_asset_storage') as mock_storage:
                mock_storage_instance = Mock()
                mock_storage_instance.get_url.return_value = "http://test.com/model.joblib"
                mock_storage.return_value = mock_storage_instance
                
                result = await context.from_estimator(model)
                assert isinstance(result, ModelRef)
                assert result.uri.startswith("memory://")  # Should be a memory URI


class TestUtilityMethods:
    """Test utility and helper methods."""
    
    def test_encode_assets_as_uri(self, context):
        """Test encoding assets as URIs recursively."""
        # Test with nested structure containing AssetRef
        asset_ref = AssetRef(data=b"test data")
        
        test_data = {
            "simple": "string",
            "asset": asset_ref,
            "nested": {
                "another_asset": asset_ref,
                "list": [asset_ref, "string"]
            },
            "tuple": (asset_ref, "value")
        }
        
        result = context.encode_assets_as_uri(test_data)
        
        assert result["simple"] == "string"
        assert isinstance(result["asset"], AssetRef)
        assert result["asset"].uri.startswith("data:application/octet-stream;base64,")
        assert isinstance(result["nested"]["another_asset"], AssetRef)
        assert result["nested"]["another_asset"].uri.startswith("data:application/octet-stream;base64,")
        assert isinstance(result["nested"]["list"][0], AssetRef)
        assert result["nested"]["list"][0].uri.startswith("data:application/octet-stream;base64,")
        assert isinstance(result["tuple"][0], AssetRef)
        assert result["tuple"][0].uri.startswith("data:application/octet-stream;base64,")
    
    def test_upload_assets_to_temp(self, context):
        """Test uploading assets to temporary storage."""
        image_ref = ImageRef(data=b"image data")
        audio_ref = AudioRef(data=b"audio data")
        
        test_data = {
            "image": image_ref,
            "audio": audio_ref,
            "normal": "string"
        }
        
        with patch.object(Environment, 'get_temp_storage') as mock_storage:
            mock_storage_instance = Mock()
            mock_storage_instance.upload_sync.return_value = "http://temp.com/uploaded_file"
            mock_storage.return_value = mock_storage_instance
            
            result = context.upload_assets_to_temp(test_data)
            
            assert result["normal"] == "string"
            assert isinstance(result["image"], ImageRef)
            assert result["image"].uri == "http://temp.com/uploaded_file"
            assert isinstance(result["audio"], AudioRef)
            assert result["audio"].uri == "http://temp.com/uploaded_file"
    
    def test_get_system_font_path(self, context):
        """Test getting system font path."""
        with patch('platform.system', return_value='Darwin'):  # macOS
            with patch('os.path.exists', return_value=True):
                with patch('os.walk') as mock_walk:
                    mock_walk.return_value = [('/Library/Fonts', [], ['Arial.ttf'])]
                    
                    result = context.get_system_font_path("Arial.ttf")
                    assert result == "/Library/Fonts/Arial.ttf"
    
    def test_get_system_font_path_not_found(self, context):
        """Test getting system font path when font not found."""
        with patch('platform.system', return_value='Linux'):
            with patch('os.path.exists', return_value=False):
                with pytest.raises(FileNotFoundError):
                    context.get_system_font_path("NonExistentFont.ttf")
    
    def test_resolve_workspace_path(self, context):
        """Test resolving workspace paths."""
        # Test workspace prefix removal
        result = context.resolve_workspace_path("/workspace/output/file.txt")
        assert result.endswith("output/file.txt")
        
        # Test relative path
        result = context.resolve_workspace_path("output/file.txt")
        assert result.endswith("output/file.txt")
    
    def test_resolve_workspace_path_traversal_protection(self, context):
        """Test workspace path traversal protection."""
        with pytest.raises(ValueError, match="outside the workspace directory"):
            context.resolve_workspace_path("../../../etc/passwd")


class TestBrowserMethods:
    """Test browser-related methods."""
    
    @pytest.mark.asyncio
    async def test_get_browser_local(self, context):
        """Test getting local browser instance."""
        with patch.object(Environment, 'get', return_value=None):  # No BROWSER_URL
            with patch('nodetool.workflows.processing_context.async_playwright') as mock_playwright:
                mock_playwright_instance = Mock()
                mock_playwright_instance.start = AsyncMock(return_value=mock_playwright_instance)
                mock_playwright_instance.chromium.launch = AsyncMock(return_value="mock_browser")
                mock_playwright.return_value = mock_playwright_instance
                
                browser = await context.get_browser()
                assert browser == "mock_browser"
                assert hasattr(context, '_browser')
    
    @pytest.mark.asyncio
    async def test_get_browser_context(self, context):
        """Test getting browser context."""
        mock_browser = Mock()
        mock_browser.new_context = AsyncMock(return_value="mock_context")
        
        with patch.object(context, 'get_browser', return_value=mock_browser):
            browser_context = await context.get_browser_context()
            assert browser_context == "mock_context"
            assert hasattr(context, '_browser_context')
    
    @pytest.mark.asyncio
    async def test_get_browser_page(self, context):
        """Test getting browser page."""
        mock_context = Mock()
        mock_page = Mock()
        mock_page.goto = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        
        with patch.object(context, 'get_browser_context', return_value=mock_context):
            page = await context.get_browser_page("http://example.com")
            assert page == mock_page
            mock_page.goto.assert_called_once_with(
                "http://example.com", 
                wait_until="domcontentloaded", 
                timeout=30000
            )
    
    @pytest.mark.asyncio
    async def test_cleanup_browser(self, context):
        """Test browser cleanup."""
        mock_browser = Mock()
        mock_browser.close = AsyncMock()
        context._browser = mock_browser
        
        await context.cleanup()
        mock_browser.close.assert_called_once()
        assert not hasattr(context, '_browser') or context._browser is None