"""
Tests for ProcessingContext audio stream helper methods.
"""
import pytest
import numpy as np
from io import BytesIO
from unittest.mock import Mock, patch, AsyncMock

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import AudioStream


@pytest.fixture
def context():
    """Create a test processing context."""
    return ProcessingContext()


class TestAudioStreamHelpers:
    """Test audio stream helper methods in ProcessingContext."""

    @pytest.mark.asyncio
    async def test_audio_stream_from_numpy_int16(self, context: ProcessingContext):
        """Test creating AudioStream from int16 numpy array."""
        # Create a simple sine wave
        duration = 0.1  # 0.1 seconds
        sample_rate = 44100
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        
        stream = await context.audio_stream_from_numpy(
            audio_data,
            sample_rate=sample_rate,
            channels=1,
        )
        
        assert stream.type == "audio_stream"
        assert stream.sample_rate == sample_rate
        assert stream.channels == 1
        assert stream.sample_width == 2
        assert stream.format == "pcm_s16le"
        assert len(stream.data) == len(audio_data) * 2  # 2 bytes per int16 sample

    @pytest.mark.asyncio
    async def test_audio_stream_from_numpy_float32(self, context: ProcessingContext):
        """Test creating AudioStream from float32 numpy array."""
        # Create a simple sine wave
        duration = 0.1
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        stream = await context.audio_stream_from_numpy(
            audio_data,
            sample_rate=sample_rate,
            channels=1,
        )
        
        assert stream.type == "audio_stream"
        assert stream.sample_rate == sample_rate
        assert stream.channels == 1
        assert stream.sample_width == 2
        assert stream.format == "pcm_s16le"
        # Data should be converted to int16
        assert len(stream.data) == len(audio_data) * 2

    @pytest.mark.asyncio
    async def test_audio_stream_from_numpy_with_timestamp(self, context: ProcessingContext):
        """Test creating AudioStream with timestamp."""
        audio_data = np.zeros(1000, dtype=np.int16)
        timestamp = (1.5, 2.5)
        
        stream = await context.audio_stream_from_numpy(
            audio_data,
            sample_rate=44100,
            channels=1,
            timestamp=timestamp,
        )
        
        assert stream.timestamp == timestamp

    @pytest.mark.asyncio
    async def test_audio_stream_from_numpy_with_metadata(self, context: ProcessingContext):
        """Test creating AudioStream with metadata."""
        audio_data = np.zeros(1000, dtype=np.int16)
        metadata = {"source": "test", "device": "default"}
        
        stream = await context.audio_stream_from_numpy(
            audio_data,
            sample_rate=44100,
            channels=1,
            metadata=metadata,
        )
        
        assert stream.metadata == metadata

    @pytest.mark.asyncio
    async def test_audio_stream_from_segment(self, context: ProcessingContext):
        """Test creating AudioStream from AudioSegment."""
        with patch("pydub.AudioSegment") as MockAudioSegment:
            # Create a mock AudioSegment
            mock_segment = Mock()
            mock_segment.raw_data = b"\x00\x01" * 1000
            mock_segment.frame_rate = 44100
            mock_segment.channels = 1
            mock_segment.sample_width = 2
            
            stream = await context.audio_stream_from_segment(
                mock_segment,
                timestamp=(0.0, 1.0),
            )
            
            assert stream.type == "audio_stream"
            assert stream.data == mock_segment.raw_data
            assert stream.sample_rate == 44100
            assert stream.channels == 1
            assert stream.sample_width == 2
            assert stream.timestamp == (0.0, 1.0)

    @pytest.mark.asyncio
    async def test_audio_stream_to_numpy_int16(self, context: ProcessingContext):
        """Test converting AudioStream to numpy array (int16)."""
        # Create an AudioStream
        num_samples = 1000
        audio_data = np.random.randint(-32768, 32767, num_samples, dtype=np.int16)
        stream = AudioStream(
            data=audio_data.tobytes(),
            sample_rate=44100,
            channels=1,
            sample_width=2,
            format="pcm_s16le",
        )
        
        # Convert to numpy
        result = await context.audio_stream_to_numpy(stream, dtype="int16")
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int16
        assert len(result) == num_samples
        np.testing.assert_array_equal(result, audio_data)

    @pytest.mark.asyncio
    async def test_audio_stream_to_numpy_float32(self, context: ProcessingContext):
        """Test converting AudioStream to numpy array (float32)."""
        # Create an AudioStream with int16 data
        num_samples = 1000
        audio_data = np.array([0, 16384, 32767, -16384, -32768], dtype=np.int16)
        stream = AudioStream(
            data=audio_data.tobytes(),
            sample_rate=44100,
            channels=1,
            sample_width=2,
            format="pcm_s16le",
        )
        
        # Convert to float32
        result = await context.audio_stream_to_numpy(stream, dtype="float32")
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == len(audio_data)
        # Check that values are in expected float range [-1.0, 1.0]
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    @pytest.mark.asyncio
    async def test_audio_stream_to_segment(self, context: ProcessingContext):
        """Test converting AudioStream to AudioSegment."""
        with patch("pydub.AudioSegment") as MockAudioSegment:
            # Create an AudioStream
            audio_data = b"\x00\x01" * 1000
            stream = AudioStream(
                data=audio_data,
                sample_rate=44100,
                channels=1,
                sample_width=2,
                format="pcm_s16le",
            )
            
            # Mock AudioSegment constructor
            mock_segment = Mock()
            MockAudioSegment.return_value = mock_segment
            
            result = await context.audio_stream_to_segment(stream)
            
            # Verify AudioSegment was constructed with correct parameters
            MockAudioSegment.assert_called_once_with(
                data=audio_data,
                sample_width=2,
                frame_rate=44100,
                channels=1,
            )
            assert result == mock_segment

    @pytest.mark.asyncio
    async def test_audio_stream_roundtrip_numpy(self, context: ProcessingContext):
        """Test roundtrip conversion: numpy -> AudioStream -> numpy."""
        # Create original numpy array
        original = np.array([0.0, 0.5, 1.0, -0.5, -1.0], dtype=np.float32)
        
        # Convert to AudioStream
        stream = await context.audio_stream_from_numpy(
            original,
            sample_rate=44100,
            channels=1,
        )
        
        # Convert back to numpy
        result = await context.audio_stream_to_numpy(stream, dtype="float32")
        
        # Check that values are approximately equal (some precision loss expected)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(original)
        np.testing.assert_allclose(result, original, rtol=0.01, atol=0.01)

    @pytest.mark.asyncio
    async def test_audio_stream_from_numpy_invalid_dtype(self, context: ProcessingContext):
        """Test that invalid dtype raises an error."""
        audio_data = np.zeros(1000, dtype=np.uint8)  # Unsupported dtype
        
        with pytest.raises(ValueError, match="Unsupported dtype"):
            await context.audio_stream_from_numpy(
                audio_data,
                sample_rate=44100,
                channels=1,
            )

    @pytest.mark.asyncio
    async def test_audio_stream_to_numpy_invalid_sample_width(self, context: ProcessingContext):
        """Test that invalid sample_width raises an error."""
        stream = AudioStream(
            data=b"\x00\x01" * 100,
            sample_rate=44100,
            channels=1,
            sample_width=3,  # Invalid sample width
            format="pcm_s24le",
        )
        
        with pytest.raises(ValueError, match="Unsupported sample_width"):
            await context.audio_stream_to_numpy(stream, dtype="float32")
