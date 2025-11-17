import asyncio
import json
import os
import signal
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any

import msgpack
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class WebSocketMode(str, Enum):
    BINARY = "binary"
    TEXT = "text"


@dataclass
class TerminalPlatform:
    is_windows: bool
    shell_cmd: list[str]


class TerminalWebSocketRunner:
    """Bridges a host shell to a WebSocket client."""

    def __init__(
        self,
        auth_token: str | None = None,
        user_id: str | None = None,
    ):
        self.auth_token = auth_token or ""
        self.user_id = user_id or ""
        self.websocket: WebSocket | None = None
        self.mode: WebSocketMode = WebSocketMode.BINARY
        self.process: asyncio.subprocess.Process | None = None
        self.master_fd: int | None = None
        self.output_task: asyncio.Task | None = None
        self.receive_task: asyncio.Task | None = None
        self.exit_task: asyncio.Task | None = None
        self._last_resize: tuple[int, int] | None = None

    @staticmethod
    def is_enabled() -> bool:
        """Feature flag for the terminal endpoint (not surfaced via settings).

        Reads directly from os.environ to avoid DEFAULT_ENV fallback, allowing tests
        to properly disable the feature via monkeypatch.
        """
        import os  # lazy import to avoid cycles

        value = os.environ.get("NODETOOL_ENABLE_TERMINAL_WS", "0")
        return value not in ("", "0", "false", "False", "no", "NO")

    def _detect_platform(self) -> TerminalPlatform:
        if os.name == "nt":
            shell = os.environ.get("NODETOOL_TERMINAL_SHELL", "powershell.exe")
            return TerminalPlatform(is_windows=True, shell_cmd=[shell, "-NoLogo"])
        shell = os.environ.get("SHELL", "/bin/bash")
        return TerminalPlatform(is_windows=False, shell_cmd=[shell, "-l"])

    async def _spawn_shell(self) -> bool:
        platform_info = self._detect_platform()
        env = os.environ.copy()

        if platform_info.is_windows:
            try:
                self.process = await asyncio.create_subprocess_exec(
                    *platform_info.shell_cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                )
            except Exception as exc:
                log.error("Failed to start Windows terminal session", exc_info=exc)
                return False
            log.info(
                "Started Windows terminal session", extra={"pid": self.process.pid}
            )
            return True

        import pty

        try:
            master_fd, slave_fd = pty.openpty()
        except OSError as exc:
            log.warning(
                "Unable to allocate PTY, falling back to pipes",
                extra={"error": str(exc), "shell": platform_info.shell_cmd},
            )
            self.master_fd = None
            pipe_shell = os.environ.get("NODETOOL_PIPE_SHELL", "/bin/sh")
            pipe_cmd = [pipe_shell]
            try:
                self.process = await asyncio.create_subprocess_exec(
                    *pipe_cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                )
            except Exception as pipe_exc:
                log.error("Failed to start pipe-based shell", exc_info=pipe_exc)
                return False
            log.info(
                "Started POSIX terminal session without PTY",
                extra={"pid": self.process.pid, "shell": pipe_cmd},
            )
            return True

        self.master_fd = master_fd
        # Ensure non-blocking reads on the PTY master
        os.set_blocking(master_fd, False)

        try:
            self.process = await asyncio.create_subprocess_exec(
                *platform_info.shell_cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env,
                start_new_session=True,
            )
        except Exception as exc:
            log.error("Failed to start PTY shell session", exc_info=exc)
            os.close(master_fd)
            os.close(slave_fd)
            self.master_fd = None
            return False
        os.close(slave_fd)
        log.info(
            "Started POSIX terminal session",
            extra={"pid": self.process.pid, "shell": platform_info.shell_cmd},
        )
        return True

    async def run(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        await websocket.accept()

        started = await self._spawn_shell()
        if not started or not self.process:
            await self._safe_close(code=1011, reason="Shell failed to start")
            return

        self.output_task = asyncio.create_task(self._forward_output())
        self.receive_task = asyncio.create_task(self._receive_messages())
        self.exit_task = asyncio.create_task(self._wait_for_exit())

        tasks = [self.receive_task, self.exit_task]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            await self.disconnect()

    async def _wait_for_exit(self) -> None:
        assert self.process is not None
        code = await self.process.wait()
        log.debug("Terminal process exited", extra={"code": code})
        await self._send_message({"type": "exit", "code": code})

    async def _forward_output(self) -> None:
        if not self.process:
            return

        if self.master_fd is not None:
            # POSIX PTY read loop
            while True:
                try:
                    chunk: bytes = await asyncio.to_thread(os.read, self.master_fd, 4096)
                except BlockingIOError:
                    await asyncio.sleep(0.01)
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                log.debug(
                    "Forwarding PTY output chunk",
                    extra={"bytes": len(chunk)},
                )
                await self._send_message({"type": "output", "data": chunk.decode("utf-8", errors="replace")})
            return

        # Windows (pipes)
        stdout = self.process.stdout
        if not stdout:
            return
        while True:
            chunk = await stdout.read(4096)
            if not chunk:
                break
            log.debug(
                "Forwarding pipe output chunk",
                extra={"bytes": len(chunk)},
            )
            await self._send_message({"type": "output", "data": chunk.decode("utf-8", errors="replace")})

    async def _receive_messages(self) -> None:
        assert self.websocket is not None
        while True:
            try:
                message = await self.websocket.receive()
            except Exception as exc:
                log.debug(f"WebSocket receive failed: {exc}")
                break

            if message.get("type") == "websocket.disconnect":
                break

            data: Any | None = None
            if "bytes" in message and message["bytes"] is not None:
                try:
                    data = msgpack.unpackb(message["bytes"])
                    self.mode = WebSocketMode.BINARY
                except Exception as exc:
                    log.warning(f"Failed to unpack msgpack message: {exc}")
                    continue
            elif "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                    self.mode = WebSocketMode.TEXT
                except Exception as exc:
                    log.warning(f"Failed to decode JSON message: {exc}")
                    continue

            if not isinstance(data, dict):
                continue

            msg_type = data.get("type")
            if msg_type == "input":
                log.debug("Terminal received input", extra={"length": len(str(data.get('data', '')))})
                await self._handle_input(str(data.get("data", "")))
            elif msg_type == "resize":
                cols = int(data.get("cols", 0) or 0)
                rows = int(data.get("rows", 0) or 0)
                await self._handle_resize(cols, rows)
            elif msg_type == "ping":
                await self._send_message({"type": "pong", "ts": asyncio.get_event_loop().time()})
            else:
                await self._send_message({"type": "error", "message": "Unknown message type"})

    async def _handle_input(self, text: str) -> None:
        if not self.process:
            return
        data = text.encode()
        if self.master_fd is not None:
            try:
                await asyncio.to_thread(os.write, self.master_fd, data)
            except Exception as exc:
                log.debug(f"Failed to write to PTY: {exc}")
            return

        stdin = self.process.stdin
        if stdin is None:
            return
        try:
            stdin.write(data)
            await stdin.drain()
        except Exception as exc:
            log.debug(f"Failed to write to stdin: {exc}")

    async def _handle_resize(self, cols: int, rows: int) -> None:
        if cols <= 0 or rows <= 0:
            return

        last = self._last_resize
        if last and last == (cols, rows):
            return
        self._last_resize = (cols, rows)

        if self.master_fd is None:
            # Windows pipes: ignore resize gracefully
            return

        try:
            import fcntl
            import struct
            import termios

            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
        except Exception as exc:
            log.debug(f"Failed to resize PTY: {exc}")

    async def _send_message(self, payload: dict[str, Any]) -> None:
        if not self.websocket:
            return
        if self.websocket.client_state == WebSocketState.DISCONNECTED:
            return
        try:
            if self.mode == WebSocketMode.BINARY:
                packed = msgpack.packb(payload, use_bin_type=True)
                await self.websocket.send_bytes(packed)
            else:
                await self.websocket.send_text(json.dumps(payload))
        except Exception as exc:
            log.debug(f"Failed to send terminal message: {exc}")

    async def _safe_close(self, code: int, reason: str) -> None:
        if self.websocket and self.websocket.client_state != WebSocketState.DISCONNECTED:
            with suppress(Exception):
                await self.websocket.close(code=code, reason=reason)

    async def disconnect(self) -> None:
        if self.receive_task and not self.receive_task.done():
            self.receive_task.cancel()
        if self.output_task and not self.output_task.done():
            self.output_task.cancel()
        if self.exit_task and not self.exit_task.done():
            self.exit_task.cancel()
        self._last_resize = None

        if self.process and self.process.returncode is None:
            with suppress(Exception):
                if os.name == "nt":
                    self.process.terminate()
                else:
                    self.process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), timeout=2)
            with suppress(Exception):
                self.process.kill()

        if self.master_fd is not None:
            with suppress(Exception):
                os.close(self.master_fd)
            self.master_fd = None

        await self._safe_close(code=1000, reason="Terminal session closed")

        self.process = None
        self.websocket = None
        self.output_task = None
        self.receive_task = None
        self.exit_task = None
