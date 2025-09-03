"""
Tests for video utility functions.
"""

import numpy as np
import pytest

from nodetool.common.video_utils import export_to_video_bytes


class TestExportToVideoBytes:
    """Test the export_to_video_bytes function."""

    def test_export_to_video_bytes_basic(self):
        """Test basic video export to bytes."""
        # Create simple test frames (2 frames, 10x10 RGB)
        frame1 = np.random.rand(10, 10, 3).astype(np.float32)
        frame2 = np.random.rand(10, 10, 3).astype(np.float32)
        video_frames = [frame1, frame2]

        # Export to bytes
        video_bytes = export_to_video_bytes(video_frames, fps=10)

        # Verify we got some bytes back
        assert isinstance(video_bytes, bytes)
        assert len(video_bytes) > 0

    def test_export_to_video_bytes_with_pil_images(self):
        """Test video export with PIL images."""
        from PIL import Image

        # Create simple PIL images
        img1 = Image.new("RGB", (10, 10), color="red")
        img2 = Image.new("RGB", (10, 10), color="blue")
        video_frames = [img1, img2]

        # Export to bytes
        video_bytes = export_to_video_bytes(video_frames, fps=10)

        # Verify we got some bytes back
        assert isinstance(video_bytes, bytes)
        assert len(video_bytes) > 0

    def test_export_to_video_bytes_custom_fps(self):
        """Test video export with custom FPS."""
        # Create simple test frames
        frame1 = np.random.rand(10, 10, 3).astype(np.float32)
        frame2 = np.random.rand(10, 10, 3).astype(np.float32)
        video_frames = [frame1, frame2]

        # Export with custom FPS
        video_bytes = export_to_video_bytes(video_frames, fps=30)

        # Verify we got some bytes back
        assert isinstance(video_bytes, bytes)
        assert len(video_bytes) > 0

    def test_export_to_video_bytes_empty_frames_raises_error(self):
        """Test that empty frame list raises an error."""
        with pytest.raises(IndexError):
            export_to_video_bytes([], fps=10)

    def test_export_to_video_bytes_single_frame(self):
        """Test video export with single frame."""
        # Create single test frame
        frame = np.random.rand(10, 10, 3).astype(np.float32)
        video_frames = [frame]

        # Export to bytes
        video_bytes = export_to_video_bytes(video_frames, fps=10)

        # Verify we got some bytes back
        assert isinstance(video_bytes, bytes)
        assert len(video_bytes) > 0


