"""
Vendorized video export utilities from diffusers.
"""

import io
import tempfile
from typing import IO, Any, List, Optional, cast

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
    video_frames: List[np.ndarray] | List[PIL.Image.Image],
    output_video_path: str | None = None,
    fps: int = 10,
) -> str:
    """Legacy video export using OpenCV backend."""
    if not _is_opencv_available():
        raise ImportError(
            "OpenCV is required for video export. Please install opencv-python: `pip install opencv-python`"
        )

    import cv2

    if output_video_path is None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            output_video_path = tmp_file.name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]  # type: ignore[union-attr]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    h, w, _c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))

    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

    video_writer.release()
    return output_video_path


def export_to_video(
    video_frames: List[np.ndarray] | List[PIL.Image.Image],
    output_video_path: str | None = None,
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
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            output_video_path = tmp_file.name

    # Convert frames to uint8 format
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]  # type: ignore[union-attr]
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
            writer.append_data(frame)  # type: ignore[attr-defined]

    return output_video_path


def export_to_video_bytes(
    video_frames: List[np.ndarray] | List[PIL.Image.Image],
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
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]  # type: ignore[union-attr]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    # Export using imageio to bytes
    from io import BytesIO

    buffer = BytesIO()
    with imageio.get_writer(
        buffer,
        format="mp4",  # type: ignore[arg-type]
        fps=fps,
        quality=quality,
        bitrate=bitrate,
        macro_block_size=macro_block_size,
    ) as writer:
        for frame in video_frames:
            writer.append_data(frame)  # type: ignore[attr-defined]

    return buffer.getvalue()


def _legacy_export_to_video_bytes(
    video_frames: List[np.ndarray] | List[PIL.Image.Image],
    fps: int = 10,
) -> bytes:
    """Legacy video export to bytes using OpenCV backend."""
    if not _is_opencv_available():
        raise ImportError(
            "OpenCV is required for video export. Please install opencv-python: `pip install opencv-python`"
        )

    import cv2

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]  # type: ignore[union-attr]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    h, w, _c = video_frames[0].shape
    video_writer = cv2.VideoWriter("/tmp/temp_video.mp4", fourcc, fps=fps, frameSize=(w, h))

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


def extract_video_frames(
    input_video: str | bytes,
    fps: int = 1,
) -> List[PIL.Image.Image]:
    """
    Extract frames from a video at a specific fps.

    Args:
        input_video: Path to video file or video bytes
        fps: Frames per second to sample. Default is 1.

    Returns:
        List[PIL.Image.Image]: List of PIL images
    """
    if not _is_imageio_available():
        if not _is_opencv_available():
            raise ImportError(
                "imageio or OpenCV is required for video reading. Please install imageio: `pip install imageio imageio-ffmpeg`"
            )
        # TODO: Implement cv2 fallback for reading
        import logging

        logging.warning("imageio not found, cv2 fallback for reading not fully implemented except for files")
        return _legacy_read_video_frames(input_video, fps)

    import imageio

    # Check for ffmpeg availability for imageio
    try:
        imageio.plugins.ffmpeg.get_exe()
    except (AttributeError, RuntimeError):
        return _legacy_read_video_frames(input_video, fps)

    frames = []

    # Handle bytes - create a BytesIO for imageio
    video_source = io.BytesIO(input_video) if isinstance(input_video, bytes) else input_video

    try:
        reader = imageio.get_reader(video_source, format="ffmpeg")  # type: ignore[arg-type]
        meta = reader.get_meta_data()
        video_fps = meta.get("fps", 30)

        # Calculate sampling interval
        step = max(1, int(video_fps / fps))

        for i, frame in enumerate(reader):  # type: ignore[arg-type]
            if i % step == 0:
                frames.append(PIL.Image.fromarray(frame))

        reader.close()
    except Exception as e:
        import logging

        logging.error(f"Error reading video with imageio: {e}")
        # Try fallback
        return _legacy_read_video_frames(input_video, fps)

    return frames


def _legacy_read_video_frames(
    input_video: str | bytes | Any,
    fps: int = 1,
) -> List[PIL.Image.Image]:
    """Legacy video reading using OpenCV."""
    if not _is_opencv_available():
        raise ImportError("OpenCV is required for video reading fallback.")

    import os
    import tempfile

    import cv2

    video_path = input_video
    temp_file = None

    # If bytes or file-like, save to temp file because cv2 needs a path
    if isinstance(input_video, (bytes, io.BytesIO)) or hasattr(input_video, "read"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            if isinstance(input_video, bytes):
                tmp.write(input_video)
            elif isinstance(input_video, io.BytesIO):
                tmp.write(input_video.getvalue())
            elif hasattr(input_video, "read"):
                file_like = cast("IO[bytes]", input_video)
                if hasattr(file_like, "getvalue"):
                    tmp.write(file_like.getvalue())  # type: ignore[attr-defined]
                else:
                    tmp.write(file_like.read())
            temp_file = tmp.name
        video_path = temp_file

    frames = []
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # Default if unknown

        step = max(1, int(video_fps / fps))

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % step == 0:
                # CV2 is BGR, convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(PIL.Image.fromarray(rgb_frame))
            count += 1

        cap.release()
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass

    return frames
