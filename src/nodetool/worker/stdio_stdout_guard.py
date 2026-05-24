"""Protect stdio worker binary protocol stream from accidental stdout writes."""

from __future__ import annotations

import sys
from typing import BinaryIO, TextIO


class _StdioStdoutGuard:
    """Route text writes to stderr while preserving binary stdout for framing."""

    def __init__(self, binary_stream: BinaryIO, text_stream: TextIO) -> None:
        self._binary_stream = binary_stream
        self._text = text_stream

    @property
    def buffer(self) -> BinaryIO:
        return self._binary_stream

    def write(self, data: str) -> int:
        if not data:
            return 0
        return self._text.write(data)

    def flush(self) -> None:
        self._text.flush()

    def isatty(self) -> bool:
        return self._text.isatty()

    def fileno(self) -> int:
        return self._binary_stream.fileno()


def install_stdio_stdout_guard() -> None:
    """Redirect library print()/stdout text to stderr; keep stdout.buffer for msgpack."""
    real_stdout = sys.__stdout__
    real_stderr = sys.__stderr__
    if real_stdout is None or real_stderr is None:
        return
    sys.stdout = _StdioStdoutGuard(real_stdout.buffer, real_stderr)  # type: ignore[assignment]
