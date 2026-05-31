"""Protect stdio worker binary protocol stream from accidental stdout writes."""

from __future__ import annotations

import os
import sys
from typing import BinaryIO, cast

_protocol_stdout_fd: int | None = None


class _ProtocolStdoutBuffer:
    """BinaryIO wrapper around the duplicated stdout fd used for msgpack frames."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def write(self, data: bytes | bytearray | memoryview) -> int:
        if isinstance(data, memoryview):
            data = data.tobytes()
        return os.write(self._fd, data)

    def flush(self) -> None:
        # os.write goes straight to the pipe; nothing to flush.
        return None

    def fileno(self) -> int:
        return self._fd


def get_protocol_stdout_buffer() -> BinaryIO:
    """Return the binary stream used exclusively for length-prefixed bridge frames."""
    if _protocol_stdout_fd is None:
        raise RuntimeError(
            "Protocol stdout is not initialized; call install_stdio_stdout_guard() first"
        )
    return cast(BinaryIO, _ProtocolStdoutBuffer(_protocol_stdout_fd))


def install_stdio_stdout_guard() -> None:
    """Redirect fd 1 (stdout) to stderr; keep a dup fd for the msgpack protocol only.

    Python ``print()`` and C extensions that write to stdout (fd 1) — including tqdm,
    safetensors, and torch/nunchaku native code — would corrupt the length-prefixed
    msgpack stream on the stdout pipe.  We dup the original stdout pipe before
    redirecting fd 1 to stderr, then write bridge frames only through the dup.
    """
    global _protocol_stdout_fd

    real_stdout = sys.__stdout__
    real_stderr = sys.__stderr__
    if real_stdout is None or real_stderr is None:
        return

    if _protocol_stdout_fd is not None:
        return

    stdout_fd = real_stdout.fileno()
    stderr_fd = real_stderr.fileno()

    # Preserve the pipe Node reads for protocol frames.
    _protocol_stdout_fd = os.dup(stdout_fd)

    # Send all library stdout (fd 1), including native writes, to stderr.
    os.dup2(stderr_fd, stdout_fd)

    encoding = getattr(real_stderr, "encoding", None) or "utf-8"
    sys.stdout = os.fdopen(stdout_fd, "w", encoding=encoding, errors="replace", closefd=False)  # type: ignore[assignment]
