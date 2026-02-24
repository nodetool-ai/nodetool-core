"""
Tests for video utility functions with fallback scenarios.
"""

import os
import tempfile
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image

from nodetool.media.video.video_utils import (
    extract_video_frames,
    export_to_video,
)

class TestVideoUtilsFallback:
    """Test video utilities with mocked dependencies to force fallbacks."""

    def setup_method(self):
        """Create a temporary video file for testing."""
        # Create a simple video with 10 frames
        self.frames = []
        for i in range(10):
            # Create frames with distinct colors to verify order/content
            img = Image.new("RGB", (64, 64), color=(i * 20, 0, 0))
            self.frames.append(img)

        # Create a temp file
        self.temp_video_fd, self.temp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(self.temp_video_fd)

        # Export video using available backend (likely imageio in test env)
        # We assume export works correctly (tested elsewhere)
        export_to_video(self.frames, self.temp_video_path, fps=10)

    def teardown_method(self):
        """Clean up temporary video file."""
        if os.path.exists(self.temp_video_path):
            os.remove(self.temp_video_path)

    def test_extract_video_frames_fallback_file(self):
        """Test extracting frames using OpenCV fallback with file path."""
        # Mock _is_imageio_available to return False
        with patch("nodetool.media.video.video_utils._is_imageio_available", return_value=False):
            # Also mock _is_opencv_available to ensure we hit the fallback path we want
            with patch("nodetool.media.video.video_utils._is_opencv_available", return_value=True):
                # Extract frames
                extracted_frames = extract_video_frames(self.temp_video_path, fps=10)

                # Verify we got frames back
                assert len(extracted_frames) > 0
                # Should get roughly 10 frames
                assert abs(len(extracted_frames) - 10) <= 2

                # Check dimensions
                assert extracted_frames[0].size == (64, 64)

    def test_extract_video_frames_fallback_bytes(self):
        """Test extracting frames using OpenCV fallback with bytes."""
        # Read video into bytes
        with open(self.temp_video_path, "rb") as f:
            video_bytes = f.read()

        # Mock _is_imageio_available to return False
        with patch("nodetool.media.video.video_utils._is_imageio_available", return_value=False):
            with patch("nodetool.media.video.video_utils._is_opencv_available", return_value=True):
                # Extract frames from bytes
                extracted_frames = extract_video_frames(video_bytes, fps=10)

                # Verify we got frames back
                assert len(extracted_frames) > 0
                assert abs(len(extracted_frames) - 10) <= 2
                assert extracted_frames[0].size == (64, 64)

    def test_extract_video_frames_fallback_ffmpeg_missing(self):
        """Test extracting frames using OpenCV fallback when ffmpeg is missing."""
        # Mock _is_imageio_available to return True
        with patch("nodetool.media.video.video_utils._is_imageio_available", return_value=True):
            # Mock imageio.plugins.ffmpeg.get_exe to raise RuntimeError
            with patch("imageio.plugins.ffmpeg.get_exe", side_effect=RuntimeError("No ffmpeg")):
                with patch("nodetool.media.video.video_utils._is_opencv_available", return_value=True):
                    # Extract frames
                    extracted_frames = extract_video_frames(self.temp_video_path, fps=10)

                    # Verify we got frames back
                    assert len(extracted_frames) > 0
                    assert abs(len(extracted_frames) - 10) <= 2
