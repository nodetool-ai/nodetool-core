"""
Server Subprocess Runner (non-Docker)
=====================================

Runs a long-lived server as a local subprocess, streams stdout/stderr, and
emits a first "endpoint" message once the server is reachable. Supports
downloading a remote binary once and caching it on disk.

Key features:
- Caches a remote binary (HTTP/HTTPS/file) under a stable cache path and
  ensures it is executable.
- Allocates or uses a provided port and exposes it to the child via args and/or
  environment. Yields a ready endpoint when TCP connect succeeds.
- Streams stdout/stderr lines and posts log updates.
- Provides a `stop()` for cooperative shutdown.

Design notes:
- API mirrors `ServerDockerRunner.stream(...)` in spirit: the async generator
  yields ("endpoint", url) once ready, then emits ("stdout"|"stderr", line) as
  the process runs, and finally completes when the process exits.
- The `user_code` argument is treated as extra CLI arguments appended to the
  binary. This allows callers to pass flags dynamically per run.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import shlex
import socket
import stat
import subprocess
import threading
import time
from typing import Any, AsyncGenerator, AsyncIterator
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import LogUpdate
from nodetool.common.environment import Environment
from nodetool.common.settings import get_system_data_path


def _safe_download_to(path: Path, url: str) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 64)
            if not chunk:
                break
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _ensure_executable(p: Path) -> None:
    try:
        mode = p.stat().st_mode
        p.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    """Safely extract the entire ZIP to dest_dir (prevents Zip Slip)."""
    dest_dir = dest_dir.resolve()
    for info in zf.infolist():
        name = info.filename
        # Disallow absolute and parent-traversal paths
        if name.startswith("/") or ".." in Path(name).parts:
            continue
        target = (dest_dir / name).resolve()
        if not str(target).startswith(str(dest_dir) + os.sep):
            continue
        if name.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info, "r") as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(1024 * 64)
                if not chunk:
                    break
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())


def _cache_remote_binary(
    url: str, name: str, archive_inner_path: str | None = None
) -> Path:
    """Return a cached path for the remote binary, downloading if needed.

    When ``archive_inner_path`` is provided, ``url`` is expected to point to a
    ZIP archive and the entire archive is extracted to a cache directory. The
    specified executable inside that archive is then used. If
    ``archive_inner_path`` is not provided, the downloaded (or copied) file
    itself is used as the executable.

    The cache path is `~/.local/share/nodetool/bin/<name>` (platform specific
    via ``get_system_data_path``). If the file does not exist, it is fetched and
    marked executable.
    """
    parsed = urlparse(url)
    bin_dir = get_system_data_path("bin/")
    dst = bin_dir / name

    if not archive_inner_path and dst.exists():
        _ensure_executable(dst)
        return dst

    bin_dir.mkdir(parents=True, exist_ok=True)

    # If an inner path is provided, we must treat the source as a zip archive
    if archive_inner_path:
        # Determine source zip path (download if remote)
        if parsed.scheme in ("", "file"):
            zip_src = Path(parsed.path)
            if not zip_src.exists():
                raise FileNotFoundError(f"Local archive not found: {url}")
        else:
            # Download the archive next to the destination for caching
            zip_src = dst.with_suffix(".zip")
            _safe_download_to(zip_src, url)

        # Extract entire archive once into a stable directory and resolve the executable inside it
        extract_dir = bin_dir / f"{name}_zip"
        exe_rel = str(archive_inner_path).strip("/\\")
        exe_path = extract_dir / Path(*exe_rel.split("/"))

        if not exe_path.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_src, "r") as zf:
                # Best-effort: verify member exists by normalized name
                norm_names = {n.rstrip("/") for n in zf.namelist()}
                if exe_rel.rstrip("/") not in norm_names:
                    # Continue anyway; extraction may normalize separators differently
                    pass
                _safe_extract_zip(zf, extract_dir)

        if not exe_path.exists():
            # Fallback: some archives include a top-level directory prefix
            candidates = sorted(extract_dir.glob(f"**/{exe_rel}"))
            if candidates:
                exe_path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"Executable path '{archive_inner_path}' not found after extraction in {extract_dir}"
                )
        _ensure_executable(exe_path)
        return exe_path

    # Otherwise treat the URL/path as a direct executable
    # Guard against accidentally passing a .zip without specifying inner path
    if archive_inner_path is None and str(parsed.path).lower().endswith(".zip"):
        raise ValueError(
            "ZIP archive URL provided but no 'archive_inner_path' was specified"
        )
    if parsed.scheme in ("", "file"):
        src = Path(parsed.path)
        if not src.exists():
            raise FileNotFoundError(f"Local binary not found: {url}")
        data = src.read_bytes()
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(dst)
    else:
        _safe_download_to(dst, url)

    _ensure_executable(dst)
    return dst


class ServerSubprocessRunner:
    """Run a server process as a local subprocess and stream logs.

    Parameters:
        binary_url: Remote or local URL to the server binary. When pointing to
            a ZIP archive, also provide ``archive_executable_path`` to extract
            the specific executable from inside the archive. The executable is
            downloaded/extracted once and cached.
        archive_executable_path: Optional path of the executable inside a ZIP
            archive referenced by ``binary_url``. Required if ``binary_url`` is a
            ZIP. Example: "bin/linux-x64/server".
        args_template: List of argument tokens. Supports `{port}` placeholder.
        port: Port to bind. If None or 0, chooses a free local port and passes it via
              `{port}` in args_template and the `port_env_var` if set.
        scheme: URL scheme for the emitted endpoint (e.g., "http", "ws").
        host_ip: Host to use in the endpoint (default "127.0.0.1").
        ready_timeout_seconds: Seconds to wait for TCP readiness before erroring.
        endpointPath: Optional path suffix for the endpoint, prefixed with '/'.
        port_env_var: If set, exports the chosen port to the child via this env var name.
        timeout_seconds: Optional max lifetime. If >0, process is terminated after timeout.

    Note:
        The `user_code` passed to `stream(...)` is appended as extra CLI args
        (shell-split via shlex) after the templated `args_template`.
    """

    def __init__(
        self,
        *,
        binary_url: str,
        args_template: list[str] | None = None,
        port: int | None = None,
        scheme: str = "ws",
        host_ip: str = "127.0.0.1",
        ready_timeout_seconds: int = 15,
        endpointPath: str = "",
        port_env_var: str | None = "PORT",
        timeout_seconds: int = 0,
        cache_root: str | Path | None = None,
        archive_executable_path: str | None = None,
    ) -> None:
        self.binary_url = binary_url
        self.args_template = list(args_template or [])
        self._requested_port = int(port) if port else 0
        self.scheme = scheme
        self.host_ip = host_ip
        self.ready_timeout_seconds = ready_timeout_seconds
        if endpointPath and not endpointPath.startswith("/"):
            endpointPath = "/" + endpointPath
        self.endpointPath = endpointPath
        self.port_env_var = port_env_var
        self.timeout_seconds = timeout_seconds
        self.archive_executable_path = archive_executable_path

        self._logger = logging.getLogger(__name__)
        # Log level is controlled by env (DEBUG/NODETOOL_LOG_LEVEL)

        # Runtime / lifecycle tracking for cooperative shutdown
        self._lock = threading.Lock()
        self._active_proc: subprocess.Popen[bytes] | None = None
        self._stopped = False

    async def stream(
        self,
        user_code: str,
        env_locals: dict[str, Any],
        context: ProcessingContext,
        node: BaseNode,
        allow_dynamic_outputs: bool = True,
        stdin_stream: AsyncIterator[str] | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        worker = threading.Thread(
            target=self._run_worker,
            kwargs=dict(
                queue=queue,
                loop=loop,
                user_code=user_code or "",
                env_locals=env_locals or {},
                context=context,
                node=node,
                stdin_stream=stdin_stream,
            ),
            daemon=True,
        )
        worker.start()

        while True:
            msg = await queue.get()
            if not isinstance(msg, dict):
                continue
            t = msg.get("type")
            if t == "yield":
                slot = msg.get("slot", "stdout")
                value = msg.get("value")
                yield slot, value
            elif t == "final":
                if not msg.get("ok", False):
                    raise RuntimeError(msg.get("error", "Unknown error"))
                break

    # ---- Public stoppable lifecycle API ----
    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            proc = self._active_proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                _wait_kill(proc, 2.0)
            except Exception:
                pass

    # ---- Worker implementation ----
    def _run_worker(
        self,
        *,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        user_code: str,
        env_locals: dict[str, Any],
        context: ProcessingContext,
        node: BaseNode,
        stdin_stream: AsyncIterator[str] | None,
    ) -> None:
        command_vec: list[str] | None = None
        proc: subprocess.Popen[bytes] | None = None
        cancel_timer: threading.Timer | None = None
        try:
            # 1) Resolve binary path (download if necessary)
            binaryPath = _cache_remote_binary(
                self.binary_url, "server", self.archive_executable_path
            )

            # 2) Resolve port
            port = self._requested_port or _find_free_port()

            # 3) Build argv
            templated = [arg.format(port=port) for arg in self.args_template]
            extra = shlex.split(user_code) if user_code else []
            argv = [str(binaryPath)] + templated + extra
            command_vec = argv

            # 4) Launch process
            env = os.environ.copy()
            if self.port_env_var:
                env[self.port_env_var] = str(port)
            cwd = getattr(context, "workspace_dir", None) or os.getcwd()

            self._logger.debug("starting subprocess: argv=%s cwd=%s", argv, cwd)
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.PIPE if stdin_stream is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # line-buffered
                universal_newlines=False,
            )
            with self._lock:
                self._active_proc = proc

            # 5) Start log reader threads
            if proc.stdout is not None:
                threading.Thread(
                    target=self._reader,
                    args=(proc.stdout, "stdout", queue, loop, context, node),
                    daemon=True,
                ).start()
            if proc.stderr is not None:
                threading.Thread(
                    target=self._reader,
                    args=(proc.stderr, "stderr", queue, loop, context, node),
                    daemon=True,
                ).start()

            # 6) Forward stdin if provided
            if stdin_stream is not None and proc.stdin is not None:
                asyncio.run_coroutine_threadsafe(
                    self._feed_stdin(proc.stdin, stdin_stream), loop
                )

            # 7) Wait until server is ready, then emit endpoint
            if not _wait_for_server_ready(
                self.host_ip, port, proc, self.ready_timeout_seconds
            ):
                raise RuntimeError(
                    f"Server did not become ready on {self.host_ip}:{port}"
                )
            endpoint = f"{self.scheme}://{self.host_ip}:{port}{self.endpointPath}"
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "yield", "slot": "endpoint", "value": endpoint}),
                loop,
            )

            # 8) Optional timeout watchdog
            if self.timeout_seconds and self.timeout_seconds > 0:
                cancel_timer = threading.Timer(
                    self.timeout_seconds, lambda: _kill(proc)
                )
                cancel_timer.daemon = True
                cancel_timer.start()

            # 9) Wait for process exit
            rc = proc.wait()
            self._logger.debug("subprocess exited with code %s", rc)
            if rc != 0 and not self._stopped:
                raise RuntimeError(f"Process exited with code {rc}")

            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "final", "ok": True}), loop
            )
        except Exception as e:
            try:
                self._logger.exception(
                    "subprocess runner error for cmd=%s: %s", command_vec, e
                )
            except Exception:
                pass
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "final", "ok": False, "error": str(e)}), loop
            )
        finally:
            try:
                if cancel_timer is not None:
                    try:
                        cancel_timer.cancel()
                    except Exception:
                        pass
                if proc is not None:
                    try:
                        if proc.poll() is None:
                            _kill(proc)
                    except Exception:
                        pass
            finally:
                with self._lock:
                    self._active_proc = None

    def _reader(
        self,
        pipe,  # IO[bytes]
        slot: str,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        context: ProcessingContext,
        node: BaseNode,
    ) -> None:
        buf = b""
        try:
            while True:
                chunk = pipe.readline()
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="ignore")
                    self._emit_line(queue, loop, context, node, slot, text)
        except Exception as e:
            self._logger.debug("reader(%s) ended: %s", slot, e)

    async def _feed_stdin(
        self, w, stdin_stream: AsyncIterator[str]
    ) -> None:  # IO[bytes]
        try:
            async for data in stdin_stream:
                if not data.endswith("\n"):
                    data = data + "\n"
                b = data.encode("utf-8")
                await asyncio.to_thread(w.write, b)
                await asyncio.to_thread(w.flush)
            try:
                await asyncio.to_thread(w.close)
            except Exception:
                pass
        except Exception as e:
            self._logger.debug("stdin feeder ended: %s", e)

    def _emit_line(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        context: ProcessingContext,
        node: BaseNode,
        slot: str,
        line: str,
    ) -> None:
        if not line.endswith("\n"):
            line = line + "\n"
        asyncio.run_coroutine_threadsafe(
            queue.put({"type": "yield", "slot": slot, "value": line}), loop
        )
        try:
            sev = "info" if slot == "stdout" else "error"
            context.post_message(
                LogUpdate(
                    node_id=node.id,
                    node_name=node.get_title(),
                    content=line[:-1],
                    severity=sev,  # type: ignore[arg-type]
                )
            )
        except Exception:
            pass


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server_ready(
    host: str, port: int, proc: subprocess.Popen[bytes], timeout: float
) -> bool:
    deadline = time.time() + max(0.0, float(timeout))
    while time.time() < deadline:
        # If the process already died, abort early
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, int(port)), timeout=1.0):
                return True
        except Exception:
            time.sleep(0.2)
    return False


def _kill(proc: subprocess.Popen[bytes]) -> None:
    try:
        proc.terminate()
    except Exception:
        pass
    _wait_kill(proc, 3.0)


def _wait_kill(proc: subprocess.Popen[bytes], grace: float) -> None:
    try:
        proc.wait(timeout=max(0.1, grace))
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


if __name__ == "__main__":
    # Lightweight demo entrypoint to manually test the ZIP download + extraction
    # and server readiness using llama.cpp's prebuilt archive.
    import asyncio as _asyncio
    import argparse as _argparse
    import sys as _sys

    class _DummyContext:
        def __init__(self, workspace_dir: str | None = None) -> None:
            self.workspace_dir = workspace_dir or os.getcwd()

        def post_message(self, msg: Any) -> None:  # pragma: no cover - manual demo
            try:
                print(
                    f"[log:{getattr(msg, 'severity', 'info')}] {getattr(msg, 'content', msg)}"
                )
            except Exception:
                print(f"[log] {msg}")

    class _DummyNode:
        id = "server-test"

        def get_title(self) -> str:  # pragma: no cover - manual demo
            return "ServerSubprocessRunnerTest"

    def _parse_args():  # pragma: no cover - manual demo
        p = _argparse.ArgumentParser(
            description="Test ServerSubprocessRunner with llama.cpp zip"
        )
        p.add_argument(
            "--url",
            default="https://github.com/ggml-org/llama.cpp/releases/download/b6348/llama-b6348-bin-macos-arm64.zip",
            help="ZIP URL containing the server executable",
        )
        p.add_argument(
            "--inner",
            default="bin/llama-server",
            help="Path of the executable inside the ZIP",
        )
        p.add_argument(
            "--port",
            type=int,
            default=0,
            help="Port to use (0 selects a free port)",
        )
        p.add_argument(
            "--timeout",
            type=int,
            default=900,
            help="Seconds to wait for server readiness",
        )
        p.add_argument(
            "--scheme",
            default="http",
            help="Endpoint scheme (http or ws)",
        )
        p.add_argument(
            "--hf",
            default="ggml-org/gpt-oss-20b-GGUF",
            help="Hugging Face repo to load with -hf",
        )
        return p.parse_args()

    async def _demo():  # pragma: no cover - manual demo
        args = _parse_args()
        ctx = ProcessingContext()
        node = BaseNode(id="server-test")

        runner = ServerSubprocessRunner(
            binary_url=args.url,
            archive_executable_path=args.inner,
            args_template=["--port", "{port}", "-hf", args.hf],
            port=args.port,
            scheme=args.scheme,
            ready_timeout_seconds=args.timeout,
            endpointPath="",
            port_env_var="PORT",
            timeout_seconds=0,
        )

        print("Starting server... (Ctrl+C to stop)")
        try:
            async for slot, value in runner.stream(
                user_code="",
                env_locals={},
                context=ctx,
                node=node,
                allow_dynamic_outputs=False,
                stdin_stream=None,
            ):
                if slot == "endpoint":
                    print(f"Endpoint: {value}")
                elif slot == "stdout":
                    print(value, end="")
                elif slot == "stderr":
                    print(value, end="", file=_sys.stderr)
        except KeyboardInterrupt:
            print("\nStopping...")
            try:
                runner.stop()
            except Exception:
                pass

    _asyncio.run(_demo())
