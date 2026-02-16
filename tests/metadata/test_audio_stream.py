"""
Tests for AudioStream type.
"""
import numpy as np
import pytest

from nodetool.metadata.types import AudioStream


class TestAudioStream:
    """Test AudioStream type creation and methods."""

    def test_create_audio_stream_basic(self):
        """Test creating a basic AudioStream."""
        audio_data = b"\x00\x01\x00\x02" * 100
        stream = AudioStream(
            data=audio_data,
            sample_rate=44100,
            channels=1,
            sample_width=2,
            format="pcm_s16le",
        )

        assert stream.type == "audio_stream"
        assert stream.data == audio_data
        assert stream.sample_rate == 44100
        assert stream.channels == 1
        assert stream.sample_width == 2
        assert stream.format == "pcm_s16le"

    def test_audio_stream_with_timestamp(self):
        """Test AudioStream with timestamp."""
        stream = AudioStream(
            data=b"\x00\x01" * 100,
            sample_rate=48000,
            channels=2,
            sample_width=2,
            timestamp=(0.5, 1.5),
        )

        assert stream.timestamp == (0.5, 1.5)

    def test_audio_stream_with_metadata(self):
        """Test AudioStream with metadata."""
        metadata = {"source": "microphone", "device": "default"}
        stream = AudioStream(
            data=b"\x00\x01" * 100,
            sample_rate=44100,
            channels=1,
            sample_width=2,
            metadata=metadata,
        )

        assert stream.metadata == metadata
        assert stream.metadata["source"] == "microphone"

    def test_audio_stream_duration_calculation(self):
        """Test duration calculation for AudioStream."""
        # Create 1 second of audio at 44100 Hz, mono, 16-bit
        num_samples = 44100
        sample_width = 2
        channels = 1
        audio_data = b"\x00\x01" * num_samples

        stream = AudioStream(
            data=audio_data,
            sample_rate=44100,
            channels=channels,
            sample_width=sample_width,
        )

        duration = stream.duration_seconds()
        assert abs(duration - 1.0) < 0.001  # Should be approximately 1 second

    def test_audio_stream_duration_stereo(self):
        """Test duration calculation for stereo AudioStream."""
        # Create 0.5 second of stereo audio at 44100 Hz, 16-bit
        num_samples = 22050  # 0.5 seconds
        sample_width = 2
        channels = 2
        # For stereo, we need data for both channels
        audio_data = b"\x00\x01" * (num_samples * channels)

        stream = AudioStream(
            data=audio_data,
            sample_rate=44100,
            channels=channels,
            sample_width=sample_width,
        )

        duration = stream.duration_seconds()
        assert abs(duration - 0.5) < 0.001  # Should be approximately 0.5 seconds

    def test_audio_stream_empty_data(self):
        """Test AudioStream with empty data."""
        stream = AudioStream()

        assert stream.data == b""
        assert stream.duration_seconds() == 0.0

    def test_audio_stream_default_values(self):
        """Test AudioStream default values."""
        stream = AudioStream()

        assert stream.type == "audio_stream"
        assert stream.data == b""
        assert stream.sample_rate == 44100
        assert stream.channels == 1
        assert stream.sample_width == 2
        assert stream.format == "pcm_s16le"
        assert stream.timestamp == (0.0, 0.0)
        assert stream.metadata is None

    def test_audio_stream_to_dict(self):
        """Test AudioStream serialization."""
        stream = AudioStream(
            data=b"\x00\x01" * 100,
            sample_rate=48000,
            channels=2,
            sample_width=2,
            format="pcm_s16le",
            timestamp=(1.0, 2.0),
            metadata={"test": "value"},
        )

        data = stream.model_dump()
        assert data["type"] == "audio_stream"
        assert data["sample_rate"] == 48000
        assert data["channels"] == 2
        assert data["sample_width"] == 2
        assert data["format"] == "pcm_s16le"
        assert data["timestamp"] == (1.0, 2.0)
        assert data["metadata"] == {"test": "value"}

    def test_audio_stream_from_dict(self):
        """Test AudioStream deserialization."""
        data = {
            "type": "audio_stream",
            "data": b"\x00\x01" * 100,
            "sample_rate": 48000,
            "channels": 2,
            "sample_width": 2,
            "format": "pcm_s16le",
            "timestamp": (1.0, 2.0),
            "metadata": {"test": "value"},
        }

        stream = AudioStream(**data)
        assert stream.type == "audio_stream"
        assert stream.sample_rate == 48000
        assert stream.channels == 2
        assert stream.timestamp == (1.0, 2.0)
