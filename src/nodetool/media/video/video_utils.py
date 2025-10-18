"""
Vendorized video export utilities from diffusers.
"""

import tempfile
from typing import List, Optional, Union

import numpy as np
import PIL.Image


def _is_imageio_available() -> bool:
    """Check if imageio is available."""
    import importlib.util

    return importlib.util.find_spec("imageio") is not None


def _is_opencv_available() -> bool:
    """Check if opencv is available."""
    import importlib.util

    return importlib.util.find_spec("cv2") is not None


def _legacy_export_to_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 10,
) -> str:
    """Legacy video export using OpenCV backend."""
    if not _is_opencv_available():
        raise ImportError(
            "OpenCV is required for video export. Please install opencv-python: `pip install opencv-python`"
        )

    import cv2

    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))

    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

    video_writer.release()
    return output_video_path


def export_to_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 10,
    quality: float = 5.0,
    bitrate: Optional[int] = None,
    macro_block_size: Optional[int] = 16,
) -> str:
    """
    Exports video frames to a video file.

    Args:
        video_frames: List of video frames as numpy arrays or PIL Images
        output_video_path: Path to save the video file. If None, creates a temporary file.
        fps: Frames per second for the output video. Default is 10.
        quality: Video output quality (0-10, where 10 is highest). Default is 5.
            Uses variable bit rate. Set to None to disable variable bitrate.
            Ignored if bitrate is specified.
        bitrate: Set a constant bitrate for the video encoding. Default is None.
            If specified, overrides the quality parameter.
        macro_block_size: Size constraint for video dimensions. Width and height
            must be divisible by this number. Default is 16. Set to None or 1
            to disable.

    Returns:
        str: Path to the exported video file
    """
    # Prefer imageio backend, fallback to OpenCV
    if not _is_imageio_available():
        import logging

        logging.warning(
            "imageio not available, falling back to OpenCV backend. "
            "For better quality, install imageio: `pip install imageio imageio-ffmpeg`"
        )
        return _legacy_export_to_video(video_frames, output_video_path, fps)

    import imageio

    # Check for ffmpeg availability
    try:
        imageio.plugins.ffmpeg.get_exe()
    except (AttributeError, RuntimeError):
        import logging

        logging.warning(
            "ffmpeg not found for imageio, falling back to OpenCV backend. "
            "Install ffmpeg support: `pip install imageio-ffmpeg`"
        )
        return _legacy_export_to_video(video_frames, output_video_path, fps)

    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    # Convert frames to uint8 format
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    # Export using imageio
    with imageio.get_writer(
        output_video_path,
        fps=fps,
        quality=quality,
        bitrate=bitrate,
        macro_block_size=macro_block_size,
    ) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    return output_video_path


def export_to_video_bytes(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    fps: int = 10,
    quality: float = 5.0,
    bitrate: Optional[int] = None,
    macro_block_size: Optional[int] = 16,
) -> bytes:
    """
    Exports video frames to bytes in memory.

    Args:
        video_frames: List of video frames as numpy arrays or PIL Images
        fps: Frames per second for the output video. Default is 10.
        quality: Video output quality (0-10, where 10 is highest). Default is 5.
            Uses variable bit rate. Set to None to disable variable bitrate.
            Ignored if bitrate is specified.
        bitrate: Set a constant bitrate for the video encoding. Default is None.
            If specified, overrides the quality parameter.
        macro_block_size: Size constraint for video dimensions. Width and height
            must be divisible by this number. Default is 16. Set to None or 1
            to disable.

    Returns:
        bytes: The video data as bytes
    """
    # Prefer imageio backend, fallback to OpenCV
    if not _is_imageio_available():
        import logging

        logging.warning(
            "imageio not available, falling back to OpenCV backend. "
            "For better quality, install imageio: `pip install imageio imageio-ffmpeg`"
        )
        return _legacy_export_to_video_bytes(video_frames, fps)

    import imageio

    # Check for ffmpeg availability
    try:
        imageio.plugins.ffmpeg.get_exe()
    except (AttributeError, RuntimeError):
        import logging

        logging.warning(
            "ffmpeg not found for imageio, falling back to OpenCV backend. "
            "Install ffmpeg support: `pip install imageio-ffmpeg`"
        )
        return _legacy_export_to_video_bytes(video_frames, fps)

    # Convert frames to uint8 format
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    # Export using imageio to bytes
    from io import BytesIO

    buffer = BytesIO()
    with imageio.get_writer(
        buffer,
        format="mp4",
        fps=fps,
        quality=quality,
        bitrate=bitrate,
        macro_block_size=macro_block_size,
    ) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    return buffer.getvalue()


def _legacy_export_to_video_bytes(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    fps: int = 10,
) -> bytes:
    """Legacy video export to bytes using OpenCV backend."""
    if not _is_opencv_available():
        raise ImportError(
            "OpenCV is required for video export. Please install opencv-python: `pip install opencv-python`"
        )

    import cv2

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        "/tmp/temp_video.mp4", fourcc, fps=fps, frameSize=(w, h)
    )

    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

    video_writer.release()

    # Read the file back into bytes
    with open("/tmp/temp_video.mp4", "rb") as f:
        video_bytes = f.read()

    # Clean up temp file
    import os

    os.remove("/tmp/temp_video.mp4")

    return video_bytes
