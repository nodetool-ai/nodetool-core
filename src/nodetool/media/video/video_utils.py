"""
Vendorized video export utilities from diffusers.
"""

import io
import tempfile
from typing import IO, Any, Optional, cast

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
    video_frames: list[np.ndarray] | list[PIL.Image.Image],
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

    # Initialize writer using dimensions from the first frame
    first_frame = video_frames[0]
    if isinstance(first_frame, PIL.Image.Image):
        w, h = first_frame.size
    elif isinstance(first_frame, np.ndarray):
        h, w = first_frame.shape[:2]
    else:
        raise ValueError(f"Unsupported frame type: {type(first_frame)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))

    for frame in video_frames:
        if isinstance(frame, PIL.Image.Image):
            frame = np.array(frame)

        # Convert to uint8 if needed (assume float 0..1 if not uint8)
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(img)

    video_writer.release()
    return output_video_path


def export_to_video(
    video_frames: list[np.ndarray] | list[PIL.Image.Image],
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

    if not video_frames:
        raise IndexError("list index out of range")

    # Export using imageio
    with imageio.get_writer(
        output_video_path,
        fps=fps,
        quality=quality,
        bitrate=bitrate,
        macro_block_size=macro_block_size,
    ) as writer:
        for frame in video_frames:
            if isinstance(frame, PIL.Image.Image):
                frame = np.array(frame)

            # Convert to uint8 if needed (assume float 0..1 if not uint8)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            writer.append_data(frame)  # type: ignore[attr-defined]

    return output_video_path


def export_to_video_bytes(
    video_frames: list[np.ndarray] | list[PIL.Image.Image],
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

    if not video_frames:
        raise IndexError("list index out of range")

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
            if isinstance(frame, PIL.Image.Image):
                frame = np.array(frame)

            # Convert to uint8 if needed (assume float 0..1 if not uint8)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            writer.append_data(frame)  # type: ignore[attr-defined]

    return buffer.getvalue()


def _legacy_export_to_video_bytes(
    video_frames: list[np.ndarray] | list[PIL.Image.Image],
    fps: int = 10,
) -> bytes:
    """Legacy video export to bytes using OpenCV backend."""
    if not _is_opencv_available():
        raise ImportError(
            "OpenCV is required for video export. Please install opencv-python: `pip install opencv-python`"
        )

    import cv2

    # Initialize writer using dimensions from the first frame
    first_frame = video_frames[0]
    if isinstance(first_frame, PIL.Image.Image):
        w, h = first_frame.size
    elif isinstance(first_frame, np.ndarray):
        h, w = first_frame.shape[:2]
    else:
        raise ValueError(f"Unsupported frame type: {type(first_frame)}")

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        temp_video_path = tmp_file.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps=fps, frameSize=(w, h))

        for frame in video_frames:
            if isinstance(frame, PIL.Image.Image):
                frame = np.array(frame)

            # Convert to uint8 if needed (assume float 0..1 if not uint8)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(img)

        video_writer.release()

        # Read the file back into bytes
        with open(temp_video_path, "rb") as f:
            video_bytes = f.read()

    finally:
        # Clean up temp file
        try:
            os.remove(temp_video_path)
        except OSError:
            pass

    return video_bytes


def extract_video_frames(
    input_video: str | bytes,
    fps: int = 1,
) -> list[PIL.Image.Image]:
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
        import logging

        logging.warning(
            "imageio not available, falling back to OpenCV backend. "
            "For better quality, install imageio: `pip install imageio imageio-ffmpeg`"
        )
        return _legacy_read_video_frames(input_video, fps)

    import imageio

    # Check for ffmpeg availability for imageio
    try:
        imageio.plugins.ffmpeg.get_exe()
    except (AttributeError, RuntimeError):
        import logging

        logging.warning(
            "ffmpeg not found for imageio, falling back to OpenCV backend. "
            "Install ffmpeg support: `pip install imageio-ffmpeg`"
        )
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
) -> list[PIL.Image.Image]:
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

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Seek only if step is large enough to justify seeking overhead
        # Benchmark shows seeking is faster when skipping > ~20 frames
        should_seek = total_frames > 0 and step > 20

        def _process_frame(frame_data):
            # CV2 is BGR, convert to RGB
            rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            return PIL.Image.fromarray(rgb_frame)

        if should_seek:
            current_frame = 0
            while current_frame < total_frames:
                # Optimized seeking for supported formats
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if not ret:
                    break

                frames.append(_process_frame(frame))
                current_frame += step
        else:
            # Fallback to sequential read
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % step == 0:
                    frames.append(_process_frame(frame))
                count += 1

        cap.release()
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass

    return frames
