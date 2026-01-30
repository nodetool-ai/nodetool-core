from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
from io import BytesIO
from typing import IO, Any, Callable, ParamSpec, TypeVar

from nodetool.media.common.media_constants import DEFAULT_AUDIO_SAMPLE_RATE

_P = ParamSpec("_P")
_T = TypeVar("_T")


async def _in_thread(func: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
    return await asyncio.to_thread(func, *args, **kwargs)


def _ensure_numpy():
    import numpy as np

    return np


def _ensure_pil():
    import PIL.Image
    import PIL.ImageOps

    return PIL.Image, PIL.ImageOps


def _ensure_audio_segment():
    from pydub import AudioSegment

    return AudioSegment


def _ensure_joblib():
    import joblib

    return joblib


def _read_all_bytes(buffer: IO[bytes]) -> bytes:
    return buffer.read()


def _read_all_bytes_from_start(buffer: IO[bytes]) -> bytes:
    with suppress(Exception):
        buffer.seek(0)
    return buffer.read()


def _read_utf8(buffer: IO[bytes]) -> str:
    return _read_all_bytes(buffer).decode("utf-8")


def _b64encode_to_str(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _b64decode_to_bytes(data: str) -> bytes:
    return base64.b64decode(data)


def _read_base64(buffer: IO[bytes]) -> str:
    return _b64encode_to_str(buffer.read())


def _pil_to_png_bytes(image: Any) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pil_to_jpeg_bytes(image: Any) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _pil_to_png_bytes_with_exif(image: Any) -> bytes:
    _, PIL_ImageOps = _ensure_pil()
    fixed = PIL_ImageOps.exif_transpose(image)
    fixed = fixed if fixed is not None else image
    buffer = BytesIO()
    fixed.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _numpy_image_to_png_bytes(arr: Any) -> bytes:
    from nodetool.media.image.image_utils import numpy_to_pil_image

    img = numpy_to_pil_image(arr)
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _audio_segment_to_mp3_bytes(segment: Any) -> bytes:
    buffer = BytesIO()
    segment.export(buffer, format="mp3")
    return buffer.getvalue()


def _audio_segment_to_wav_bytes(segment: Any) -> bytes:
    buffer = BytesIO()
    segment.export(buffer, format="wav")
    return buffer.getvalue()


def _numpy_audio_to_mp3_bytes(arr: Any, sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE) -> bytes:
    np = _ensure_numpy()
    AudioSegment = _ensure_audio_segment()

    channels = 1
    audio_arr = arr
    if getattr(audio_arr, "ndim", 0) == 2:
        channels = int(audio_arr.shape[1])

    if audio_arr.dtype == np.int16:
        raw = audio_arr.tobytes()
    elif audio_arr.dtype in (np.float32, np.float64, np.float16):
        raw = (audio_arr * (2**14)).astype(np.int16).tobytes()
    else:
        raise ValueError(f"Unsupported audio ndarray dtype {audio_arr.dtype}")

    seg = AudioSegment(
        data=raw,
        frame_rate=sample_rate,
        sample_width=2,
        channels=int(channels),
    )
    return _audio_segment_to_mp3_bytes(seg)


def _numpy_video_to_mp4_bytes(video_arr: Any, fps: int = 30) -> bytes:
    from nodetool.media.video.video_utils import export_to_video_bytes as _exporter

    video_frames = list(video_arr)
    return _exporter(video_frames, fps=fps, quality=5.0, bitrate=None, macro_block_size=16)


def _open_image_as_rgb(buffer: IO[bytes]) -> Any:
    PIL_Image, PIL_ImageOps = _ensure_pil()
    with suppress(Exception):
        buffer.seek(0)
    image = PIL_Image.open(buffer)
    # Force PIL to load the image data into memory so the buffer can be closed
    # Without this, PIL keeps the buffer open, causing file descriptor leaks
    image.load()
    try:
        rotated = PIL_ImageOps.exif_transpose(image)
        image = rotated if rotated is not None else image
    except (AttributeError, KeyError, TypeError):
        pass
    return image.convert("RGB")


def _audio_segment_from_file(buffer: IO[bytes]) -> Any:
    AudioSegment = _ensure_audio_segment()
    with suppress(Exception):
        buffer.seek(0)
    return AudioSegment.from_file(buffer)


def _audio_segment_to_numpy(
    segment: Any,
    *,
    sample_rate: int,
    mono: bool,
) -> tuple[Any, int, int]:
    np = _ensure_numpy()
    segment = segment.set_frame_rate(sample_rate)
    if mono and segment.channels > 1:
        segment = segment.set_channels(1)
    samples = np.array(segment.get_array_of_samples())
    max_value = float(2 ** (8 * segment.sample_width - 1))
    samples = samples.astype(np.float32) / max_value
    return samples, segment.frame_rate, segment.channels


def _joblib_load_from_io(buffer: IO[bytes]) -> Any:
    joblib = _ensure_joblib()
    with suppress(Exception):
        buffer.seek(0)
    return joblib.load(buffer)


def _joblib_dump_to_bytes(estimator: Any) -> bytes:
    joblib = _ensure_joblib()
    stream = BytesIO()
    joblib.dump(estimator, stream)
    return stream.getvalue()

