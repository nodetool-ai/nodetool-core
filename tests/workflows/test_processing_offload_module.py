from __future__ import annotations

import threading
from io import BytesIO

import pytest


@pytest.mark.asyncio
async def test_in_thread_runs_off_event_loop_thread():
    from nodetool.workflows.processing_offload import _in_thread

    main_thread_id = threading.get_ident()
    worker_thread_id = await _in_thread(threading.get_ident)
    assert worker_thread_id != main_thread_id


def test_open_image_as_rgb_roundtrip():
    import PIL.Image

    from nodetool.workflows.processing_offload import _open_image_as_rgb, _pil_to_png_bytes

    src = PIL.Image.new("RGBA", (8, 9), (10, 20, 30, 128))
    png_bytes = _pil_to_png_bytes(src)

    img = _open_image_as_rgb(BytesIO(png_bytes))
    assert img.mode == "RGB"
    assert img.size == (8, 9)


def test_joblib_dump_load_roundtrip():
    from nodetool.workflows.processing_offload import _joblib_dump_to_bytes, _joblib_load_from_io

    payload = {"a": 1, "b": [1, 2, 3], "nested": {"ok": True}}
    dumped = _joblib_dump_to_bytes(payload)
    loaded = _joblib_load_from_io(BytesIO(dumped))
    assert loaded == payload


def _has_ffmpeg() -> bool:
    from pydub.utils import which

    return which("ffmpeg") is not None


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg is required for mp3 export/import")
def test_audio_segment_mp3_export_and_import_without_mocking():
    from pydub import AudioSegment

    from nodetool.workflows.processing_offload import _audio_segment_from_file, _audio_segment_to_mp3_bytes

    segment = AudioSegment.silent(duration=200, frame_rate=22050)
    mp3_bytes = _audio_segment_to_mp3_bytes(segment)
    assert len(mp3_bytes) > 16

    decoded = _audio_segment_from_file(BytesIO(mp3_bytes))
    assert 150 <= len(decoded) <= 500


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg is required for mp3 export/import")
def test_numpy_audio_mp3_export_and_import_without_mocking():
    import numpy as np

    from nodetool.workflows.processing_offload import _audio_segment_from_file, _numpy_audio_to_mp3_bytes

    # 0.1s of silence @ 44100Hz
    audio = np.zeros((4410,), dtype=np.float32)
    mp3_bytes = _numpy_audio_to_mp3_bytes(audio, sample_rate=44100)
    assert len(mp3_bytes) > 16

    decoded = _audio_segment_from_file(BytesIO(mp3_bytes))
    assert 50 <= len(decoded) <= 500

