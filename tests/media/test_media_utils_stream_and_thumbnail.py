"""Regression tests for media_utils stream rewinding and video thumbnail sizing."""

import io
import os
import stat
import tempfile
import wave

import pytest

from nodetool.media.common import media_utils

pytestmark = pytest.mark.asyncio


def _write_stub(tmp_path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


async def test_get_video_duration_rewinds_input(tmp_path, monkeypatch) -> None:
    """A stream already read by the caller must still be written to the temp file."""
    stub = _write_stub(
        tmp_path,
        "ffprobe_stub.sh",
        'for a in "$@"; do last="$a"; done\n'
        'if [ -s "$last" ]; then echo 1.5; exit 0; else echo empty >&2; exit 1; fi\n',
    )
    monkeypatch.setattr(media_utils, "FFPROBE_PATH", stub)

    buf = io.BytesIO(b"video bytes" * 10)
    buf.read()  # caller consumed the stream

    assert await media_utils.get_video_duration(buf) == 1.5


async def test_get_audio_duration_rewinds_input() -> None:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 8000)  # 1 second
    buf.read()  # caller consumed the stream

    assert media_utils.get_audio_duration(buf) == pytest.approx(1.0, abs=0.05)


async def test_create_video_thumbnail_scales_and_rewinds(tmp_path, monkeypatch) -> None:
    """The ffmpeg filter chain must contain a scale filter using the requested size."""
    args_file = tmp_path / "args.txt"
    stub = _write_stub(
        tmp_path,
        "ffmpeg_stub.sh",
        f'echo "$@" > {args_file}\n'
        'if [ ! -s "$2" ]; then echo "empty input" >&2; exit 1; fi\n'
        "printf THUMB\n",
    )
    monkeypatch.setattr(media_utils, "FFMPEG_PATH", stub)

    buf = io.BytesIO(b"video bytes" * 10)
    buf.read()  # caller consumed the stream

    out = await media_utils.create_video_thumbnail(buf, 120, 80)

    assert out.read() == b"THUMB"
    recorded = args_file.read_text()
    assert "scale=" in recorded
    assert "120" in recorded and "80" in recorded


async def test_create_video_thumbnail_removes_temp_file_on_error(tmp_path, monkeypatch) -> None:
    stub = _write_stub(tmp_path, "ffmpeg_fail.sh", "echo boom >&2\nexit 1\n")
    monkeypatch.setattr(media_utils, "FFMPEG_PATH", stub)

    before = set(os.listdir(tempfile.gettempdir()))
    with pytest.raises(Exception, match="ffmpeg error"):
        await media_utils.create_video_thumbnail(io.BytesIO(b"data"), 100, 100)
    after = set(os.listdir(tempfile.gettempdir()))

    assert after <= before


async def test_create_video_thumbnail_removes_temp_file_when_read_fails(tmp_path, monkeypatch) -> None:
    class Broken(io.BytesIO):
        def read(self, *args, **kwargs):  # type: ignore[override]
            raise OSError("stream broken")

    before = set(os.listdir(tempfile.gettempdir()))
    with pytest.raises(OSError, match="stream broken"):
        await media_utils.create_video_thumbnail(Broken(b"data"), 100, 100)
    after = set(os.listdir(tempfile.gettempdir()))

    assert after <= before
