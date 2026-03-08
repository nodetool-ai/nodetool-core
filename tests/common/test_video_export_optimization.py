"""
Tests for video export optimization and bug fix.
"""

import io
import os
import tempfile

import imageio
import numpy as np
import pytest
from PIL import Image

from nodetool.media.video.video_utils import export_to_video_bytes


class TestVideoExportOptimization:
    """Test the export_to_video_bytes function optimization and correctness."""

    def test_export_uint8_preserves_values(self):
        """Test that uint8 frames are not modified (fixing the overflow bug)."""
        # Create a frame with specific value that would overflow if multiplied by 255
        # 100 * 255 = 25500. 25500 % 256 = 156.
        val = 100
        frame = np.full((64, 64, 3), val, dtype=np.uint8)
        frames = [frame]

        # Export to bytes
        video_bytes = export_to_video_bytes(frames, fps=1)

        # Write to temp file for reading back
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            reader = imageio.get_reader(tmp_path, format="ffmpeg")
            read_frame = reader.get_data(0)
            reader.close()

            # Check mean value. Compression might add small noise.
            mean_val = np.mean(read_frame)
            assert abs(mean_val - val) < 5, f"Value mismatch: got {mean_val}, expected {val}"

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_export_float_converts_correctly(self):
        """Test that float frames (0..1) are converted correctly."""
        # 0.5 -> 127/128
        val_float = 0.5
        frame = np.full((64, 64, 3), val_float, dtype=np.float32)
        frames = [frame]

        video_bytes = export_to_video_bytes(frames, fps=1)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            reader = imageio.get_reader(tmp_path, format="ffmpeg")
            read_frame = reader.get_data(0)
            reader.close()

            mean_val = np.mean(read_frame)
            expected_val = 127.5
            assert abs(mean_val - expected_val) < 5, f"Value mismatch: got {mean_val}, expected ~{expected_val}"

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_export_pil_images(self):
        """Test that PIL images are handled correctly."""
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        frames = [img]

        video_bytes = export_to_video_bytes(frames, fps=1)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            reader = imageio.get_reader(tmp_path, format="ffmpeg")
            read_frame = reader.get_data(0)
            reader.close()

            mean_val = np.mean(read_frame)
            assert abs(mean_val - 100) < 5, f"Value mismatch: got {mean_val}, expected 100"

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_mixed_types_handled(self):
        """Test that a list with mixed types (though unlikely) works if first frame logic holds."""
        # Note: Current implementation uses first frame to determine video size.
        # If frames are same size, mixed types should work due to lazy conversion loop.

        frame1 = np.full((64, 64, 3), 100, dtype=np.uint8)
        frame2 = np.full((64, 64, 3), 0.5, dtype=np.float32)  # should become ~127
        frames = [frame1, frame2]

        video_bytes = export_to_video_bytes(frames, fps=2)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            reader = imageio.get_reader(tmp_path, format="ffmpeg")
            f1 = reader.get_data(0)
            f2 = reader.get_data(1)
            reader.close()

            assert abs(np.mean(f1) - 100) < 5
            assert abs(np.mean(f2) - 127.5) < 5

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
