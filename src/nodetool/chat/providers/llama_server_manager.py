"""
Llama.cpp server process manager.

Spawns and caches llama-server processes per model spec, keeps them alive with
an inactivity TTL, and returns a ready base URL for OpenAI-compatible access.

Model spec formats supported:
 - Absolute/relative path to a .gguf file → uses `-m <path>`
 - "<repo_id>:<filename>" (filename contains a dot) → resolves from local
   Hugging Face hub cache and uses `-m <resolved_path>`. If not present in the
   local cache, raises an error instructing to download it first (no fallback).
 - "<repo_id>:<quant_or_tag>" (no dot) → uses `-hf <repo_id>:<quant_or_tag>`
 - "<repo_id>" → uses `-hf <repo_id>` (defaults quant per server behavior)

Environment variables:
 - LLAMA_SERVER_BINARY: path/name of llama-server (default: "llama-server")
 - LLAMA_SERVER_HOST: bind host (default: 127.0.0.1)
 - LLAMA_SERVER_THREADS: optional int passed as --threads
 - LLAMA_SERVER_PARALLEL: optional int passed as --parallel
 - LLAMA_SERVER_CTX_SIZE: optional int passed as --ctx-size
 - LLAMA_SERVER_N_GPU_LAYERS: optional int passed as --n-gpu-layers
 - LLAMA_SERVER_EXTRA_ARGS: extra CLI args appended as a single string
 - LLAMA_SERVER_READY_TIMEOUT: seconds to wait for readiness (default 300)
 - LLAMA_SERVER_TTL_SECONDS: inactivity TTL before shutdown (default 300)
 - HF_TOKEN: forwarded to process via --hf-token (and used for HF cache lookups)
 - LLAMA_API_KEY: optional API key to enforce auth on server (default none)
 - HUGGINGFACE_HUB_CACHE / HF_HOME: to locate the local HF hub cache
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import shlex
import signal
import socket
import time
import atexit
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Any
import weakref

import httpx

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


def _find_free_port() -> int:
    """Find an available TCP port bound to 127.0.0.1.

    Returns:
        A free port number chosen by the operating system.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _is_windows_abs_path(path: str) -> bool:
    """Return True if the string looks like a Windows absolute or UNC path.

    Args:
        path: Path-like string to inspect.

    Returns:
        True if `path` looks like ``C:\\...`` or a UNC path starting with ``\\\\``,
        False otherwise.

    Notes:
        This helps disambiguate from HF repo specs like ``org/repo:file.gguf``,
        which contain a colon but are not filesystem paths on POSIX systems.
    """
    try:
        if len(path) >= 3 and path[1] == ":" and (path[2] == "\\" or path[2] == "/"):
            return True
        if path.startswith("\\\\"):
            return True
    except Exception:
        pass
    return False


def _is_path_model(model: str) -> bool:
    """Heuristically determine whether the model spec is a filesystem path.

    Rules:
    - POSIX abs/rel prefixes: "/", "./", "../"
    - Windows absolute or UNC paths
    - Bare filename ending with ".gguf" that does not contain a colon (to avoid
      misclassifying HF specs like "org/repo:file.gguf").

    Args:
        model: Model spec string as provided by the caller.

    Returns:
        True if `model` is recognized as a local path, False otherwise.
    """
    if model.startswith("/") or model.startswith("./") or model.startswith("../"):
        return True
    if _is_windows_abs_path(model):
        return True
    if model.lower().endswith(".gguf") and ":" not in model:
        return True
    return False


def _hf_cache_dir() -> str:
    """Return the local Hugging Face hub cache directory.

    Order of precedence:
    - HUGGINGFACE_HUB_CACHE (must exist)
    - HF_HOME/hub (must exist)
    - ~/.cache/huggingface/hub

    Returns:
        Path to the hub cache directory. The path may not exist if the cache
        has not been created yet.
    """
    cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache and os.path.isdir(cache):
        return cache
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        path = os.path.join(hf_home, "hub")
        if os.path.isdir(path):
            return path
    # Default
    return os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub"))


def _resolve_hf_cached_file(repo_id: str, filename: str) -> Optional[str]:
    """Resolve a repo file to a local path in the Hugging Face cache.

    Args:
        repo_id: Hugging Face repo id like "org/repo".
        filename: File name within the repo, e.g., "model.Q4_K_M.gguf".

    Returns:
        Absolute path to the file if found locally, otherwise None.
    """
    hub_cache = _hf_cache_dir()
    repo_dir = f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = os.path.join(hub_cache, repo_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    snapshot_paths = [
        os.path.join(snapshots_dir, d)
        for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]
    # Sort by mtime, newest first
    snapshot_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for snap in snapshot_paths:
        candidate = os.path.join(snap, filename)
        if os.path.exists(candidate):
            log.debug(
                f"Resolved HF file from cache snapshots: {repo_id}:{filename} -> {candidate}"
            )
            return candidate

    return None


def _parse_model_args(model: str) -> Tuple[list[str], str]:
    """Build llama-server model arguments from a model spec.

    Args:
        model: Model spec. Supported forms:
            - Absolute/relative .gguf file path
            - "<repo_id>:<filename>" where <filename> has a dot
            - "<repo_id>:<quant_or_tag>" with no dot
            - "<repo_id>"

    Returns:
        A tuple (args, alias) where:
        - args: List of CLI arguments for llama-server representing the model.
        - alias: A stable alias to pass via --alias.

    Raises:
        FileNotFoundError: If a "<repo_id>:<filename>" cannot be found locally.
    """
    model = model.strip()
    alias = model
    if _is_path_model(model):
        return ["-m", model], alias
    if ":" in model:
        repo_id, tail = model.split(":", 1)
        if "." in tail:
            # looks like a file name within the repo; try local HF cache first
            resolved = _resolve_hf_cached_file(repo_id, tail)
            if resolved:
                return ["-m", resolved], alias
            raise FileNotFoundError(
                (
                    "Hugging Face model file not found in local cache: "
                    f"{repo_id}:{tail}. Please download the model file locally "
                    "(e.g., via 'huggingface_hub' or 'git lfs') so it appears in your HF hub cache, "
                    "or provide an absolute path to the .gguf file."
                )
            )
        else:
            # quant or tag appended to repo
            return ["-hf", f"{repo_id}:{tail}"], alias
    # Only repo id passed
    return ["-hf", model], alias


def _gpu_seems_available() -> bool:
    """Heuristically determine whether a GPU seems available.

    Considers:
    - LLAMA_FORCE_CPU: if set, force False
    - CUDA_VISIBLE_DEVICES: non-empty and not -1/none
    - Availability of `nvidia-smi` on PATH
    - torch.cuda.is_available() if torch is importable

    Returns:
        True if GPU availability is likely, False otherwise.
    """
    try:
        if os.environ.get("LLAMA_FORCE_CPU"):
            return False
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None and cvd.strip() not in ("-1", "", "none"):
            return True
        if shutil.which("nvidia-smi") is not None:
            return True
        try:
            import torch  # type: ignore

            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                return True
        except Exception:
            pass
    except Exception:
        pass
    return False


if TYPE_CHECKING:  # pragma: no cover - typing hints only
    pass


@dataclasses.dataclass
class _RunningServer:
    """Container for a running llama-server process and its metadata.

    Attributes:
        process: Asyncio subprocess handle (or shim) for the server process.
        base_url: Base HTTP URL of the server, e.g., "http://127.0.0.1:12345".
        model_key: Key used to index this server in the manager (the model spec).
        alias: Alias advertised to the server via --alias.
        last_used: UNIX timestamp of the last access.
        host: Host bound by the server.
        port: Port bound by the server.
        stdout_task: Background task forwarding stdout logs, if any.
        stderr_task: Background task forwarding stderr logs, if any.
    """

    process: asyncio.subprocess.Process
    base_url: str
    model_key: str
    alias: str
    last_used: float
    host: str
    port: int
    stdout_task: asyncio.Task | None = None
    stderr_task: asyncio.Task | None = None


# Global registry to ensure subprocesses are terminated on interpreter exit
_GLOBAL_PIDS: set[int] = set()


def _register_pid(pid: int) -> None:
    """Register a PID for best-effort cleanup at interpreter exit.

    Args:
        pid: Process ID to track.
    """
    try:
        _GLOBAL_PIDS.add(pid)
    except Exception:
        pass


def _unregister_pid(pid: int) -> None:
    """Unregister a PID previously added to the global registry.

    Args:
        pid: Process ID to stop tracking.
    """
    try:
        _GLOBAL_PIDS.discard(pid)
    except Exception:
        pass


def _kill_pid(pid: int, sig: int) -> None:
    """Send a signal to a PID, ignoring errors if the process is gone.

    Args:
        pid: Target process ID.
        sig: Signal number to send.
    """
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return
    except Exception:
        # Best-effort; ignore
        pass


def _atexit_kill_all() -> None:  # pragma: no cover - runs at interpreter shutdown
    """Terminate all registered PIDs at interpreter shutdown.

    Tries SIGTERM first, waits briefly, then SIGKILL for any stragglers.
    All failures are ignored as this is best-effort cleanup.
    """
    pids = list(_GLOBAL_PIDS)
    if not pids:
        return
    # Try graceful first
    for pid in pids:
        _kill_pid(pid, signal.SIGTERM if hasattr(signal, "SIGTERM") else signal.SIGINT)
    try:
        time.sleep(0.5)
    except Exception:
        pass
    # Force kill any stragglers
    for pid in list(_GLOBAL_PIDS):
        _kill_pid(pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
        _unregister_pid(pid)


atexit.register(_atexit_kill_all)


class LlamaServerManager:
    """Manage lifecycle of llama-server processes keyed by model spec.

    Starts servers on demand, reuses them across calls, prunes them after an
    inactivity TTL, and exposes OpenAI-compatible base URLs.
    """

    def __init__(self, ttl_seconds: Optional[int] = None):
        """Initialize the manager.

        Args:
            ttl_seconds: Optional inactivity TTL in seconds. If None, uses
                LLAMA_SERVER_TTL_SECONDS from the environment or defaults to 300.

        Environment:
            - LLAMA_SERVER_BINARY
            - LLAMA_SERVER_HOST
            - LLAMA_SERVER_READY_TIMEOUT
            - LLAMA_SERVER_TTL_SECONDS
            - LLAMA_SERVER_THREADS
            - LLAMA_SERVER_PARALLEL
            - LLAMA_SERVER_CTX_SIZE
            - LLAMA_SERVER_N_GPU_LAYERS
            - LLAMA_SERVER_EXTRA_ARGS
            - HF_TOKEN
            - LLAMA_API_KEY

        Notes:
            If a GPU appears available and LLAMA_SERVER_N_GPU_LAYERS is not set,
            this defaults to "999" to enable maximum offload.
        """
        self._binary = Environment.get("LLAMA_SERVER_BINARY", "llama-server")
        self._host = Environment.get("LLAMA_SERVER_HOST", "127.0.0.1")
        self._ready_timeout = int(Environment.get("LLAMA_SERVER_READY_TIMEOUT", 300))
        self._ttl_seconds = int(
            Environment.get("LLAMA_SERVER_TTL_SECONDS", ttl_seconds or 300)
        )
        self._threads = Environment.get("LLAMA_SERVER_THREADS")
        self._parallel = Environment.get("LLAMA_SERVER_PARALLEL")
        self._ctx_size = Environment.get("LLAMA_SERVER_CTX_SIZE")
        self._n_gpu_layers = Environment.get("LLAMA_SERVER_N_GPU_LAYERS")
        self._extra_args = Environment.get("LLAMA_SERVER_EXTRA_ARGS", "")
        self._hf_token = Environment.get("HF_TOKEN")
        self._api_key = Environment.get("LLAMA_API_KEY")

        self._servers: Dict[str, _RunningServer] = {}
        self._lock = asyncio.Lock()
        self._pruner_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._signals_installed = False
        self._atexit_registered = False

        # Auto-enable GPU offload if available unless explicitly set
        if not self._n_gpu_layers and _gpu_seems_available():
            self._n_gpu_layers = "999"
            log.debug("GPU detected; defaulting --n-gpu-layers=999")

    async def ensure_server(self, model: str) -> str:
        """Ensure a llama-server is running for the given model and return its URL.

        If a compatible server already exists, updates its last-used timestamp and
        returns immediately; otherwise spawns a new process and waits for readiness.

        Args:
            model: Model spec string. See `_parse_model_args` for supported forms.

        Returns:
            Base URL as "http://host:port" suitable for OpenAI-compatible clients.

        Raises:
            RuntimeError: If the server fails to become ready before the timeout.
            FileNotFoundError: For "<repo_id>:<filename>" when the file is not cached.
        """
        model_key = model.strip()
        async with self._lock:
            inst = self._servers.get(model_key)
            now = time.time()
            if inst and inst.process.returncode is None:
                inst.last_used = now
                self._ensure_pruner()
                return inst.base_url

            # Capture loop and install shutdown hooks on first use
            if self._loop is None:
                try:
                    self._loop = asyncio.get_running_loop()
                except RuntimeError:
                    self._loop = None
            self._install_signal_handlers()
            self._register_instance_atexit()

            # Start a new server
            port = _find_free_port()
            model_args, alias = _parse_model_args(model_key)

            argv = [
                self._binary,
                "--host",
                self._host,
                "--port",
                str(port),
                "--no-webui",
                "--alias",
                alias,
                "--jinja",
            ] + model_args

            if self._threads:
                argv += ["--threads", str(self._threads)]
            if self._parallel:
                argv += ["--parallel", str(self._parallel)]
            if self._ctx_size:
                argv += ["--ctx-size", str(self._ctx_size)]
            if self._n_gpu_layers:
                argv += ["--n-gpu-layers", str(self._n_gpu_layers)]
            if self._hf_token:
                argv += ["--hf-token", str(self._hf_token)]
            if self._api_key:
                argv += ["--api-key", str(self._api_key)]

            if self._extra_args:
                argv += shlex.split(self._extra_args)

            def _format_argv_for_log(args: list[str]) -> str:
                """Return a loggable command string with sensitive values redacted.

                Args:
                    args: Full argv list.

                Returns:
                    A single string with tokens joined and --hf-token/--api-key values redacted.
                """
                redacted: list[str] = []
                redact_next = False
                for part in args:
                    if redact_next:
                        redacted.append("[REDACTED]")
                        redact_next = False
                        continue
                    if part in ("--hf-token", "--api-key"):
                        redacted.append(part)
                        redact_next = True
                    else:
                        redacted.append(part)
                return " ".join(shlex.quote(a) for a in redacted)

            log.debug(f"Starting llama-server: {_format_argv_for_log(argv)}")
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except NotImplementedError:
                # Windows: some event loop implementations lack subprocess support
                loop = asyncio.get_running_loop()

                def _spawn_blocking() -> tuple[int, Any]:
                    """Spawn the server using subprocess.Popen as a blocking fallback.

                    Returns:
                        A tuple of (pid, Popen instance).
                    """
                    import subprocess

                    # Use Popen as a fallback; we will poll it asynchronously
                    p = subprocess.Popen(
                        argv,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    return p.pid or -1, p

                pid, popen = await loop.run_in_executor(None, _spawn_blocking)

                class _ProcShim:
                    """Small adapter exposing a Popen-like process to asyncio callers."""

                    def __init__(self, pid: int, popen: Any):
                        self._popen = popen
                        self.pid = pid
                        self.returncode = None
                        # Expose stdout/stderr as async-friendly readers using threads
                        self.stdout = None
                        self.stderr = None

                    async def wait(self):
                        """Wait for process termination in a thread executor."""
                        loop2 = asyncio.get_running_loop()
                        return await loop2.run_in_executor(None, self._popen.wait)

                    def terminate(self):
                        """Request graceful process termination."""
                        try:
                            self._popen.terminate()
                        except Exception:
                            pass

                    def kill(self):
                        """Forcefully kill the process."""
                        try:
                            self._popen.kill()
                        except Exception:
                            pass

                proc = _ProcShim(pid, popen)  # type: ignore[assignment]
            if proc.pid is not None:
                _register_pid(proc.pid)

            base_url = f"http://{self._host}:{port}"

            # Background async readers to forward server logs to our logger
            async def _reader(stream: asyncio.StreamReader, which: str) -> None:
                """Read a process stream line-by-line and forward to the logger.

                Args:
                    stream: Async stream reader to consume.
                    which: Label used in debug messages.
                """
                try:
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        text = line.decode("utf-8", errors="ignore").rstrip("\n")
                        log.info(text)
                except Exception as e:
                    log.debug(f"log reader ended: {which}: {e}")

            t_out: asyncio.Task | None = None
            t_err: asyncio.Task | None = None
            # Only attach readers if using asyncio subprocess with streams
            if hasattr(proc, "stdout") and isinstance(
                proc.stdout, asyncio.StreamReader
            ):
                if proc.stdout is not None:
                    t_out = asyncio.create_task(_reader(proc.stdout, "stdout"))
            if hasattr(proc, "stderr") and isinstance(
                proc.stderr, asyncio.StreamReader
            ):
                if proc.stderr is not None:
                    t_err = asyncio.create_task(_reader(proc.stderr, "stderr"))

            ok = await self._wait_ready(base_url)
            if not ok:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    if proc.pid is not None:
                        _unregister_pid(proc.pid)
                except Exception:
                    pass
                raise RuntimeError("llama-server did not become ready in time")

            inst = _RunningServer(
                process=proc,
                base_url=base_url,
                model_key=model_key,
                alias=alias,
                last_used=now,
                host=self._host,
                port=port,
                stdout_task=t_out,
                stderr_task=t_err,
            )
            self._servers[model_key] = inst
            self._ensure_pruner()
            return base_url

    async def _wait_ready(self, base_url: str) -> bool:
        """Poll the server health endpoint until ready or timeout.

        Args:
            base_url: Base URL of the server to poll.

        Returns:
            True if the server responded with HTTP 200 on /health before the deadline,
            False otherwise.
        """
        deadline = time.time() + max(5, self._ready_timeout)
        async with httpx.AsyncClient(timeout=5.0, verify=False) as client:  # nosec B501
            while time.time() < deadline:
                try:
                    r = await client.get(f"{base_url}/health")
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        return False

    def _ensure_pruner(self) -> None:
        """Start the background pruning task if not already running."""
        if self._pruner_task is None or self._pruner_task.done():
            self._pruner_task = asyncio.create_task(self._prune_loop())

    def _install_signal_handlers(self) -> None:
        """Install signal handlers to trigger graceful shutdown on SIGINT/SIGTERM."""
        if self._signals_installed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        # Install best-effort handlers; not all platforms allow this
        def _handle_signal(signame: str) -> None:
            try:
                # Avoid re-entrancy
                if getattr(self, "_shutdown_started", False):
                    return
                setattr(self, "_shutdown_started", True)
            except Exception:
                pass
            try:
                loop.create_task(self.stop_all())
            except Exception:
                self.shutdown_sync()

        for sig_name in ("SIGINT", "SIGTERM"):
            sig_obj = getattr(signal, sig_name, None)
            if sig_obj is None:
                continue
            try:
                loop.add_signal_handler(sig_obj, _handle_signal, sig_name)
            except (NotImplementedError, RuntimeError):
                # Fallback to basic signal module for non-asyncio environments
                try:
                    signal.signal(sig_obj, lambda *_: _handle_signal(sig_name))
                except Exception:
                    pass
        self._signals_installed = True

    def _register_instance_atexit(self) -> None:
        """Register a per-instance atexit hook to stop all servers on exit."""
        if self._atexit_registered:
            return
        mgr_ref = weakref.ref(self)

        def _cleanup_instance() -> None:  # pragma: no cover - exit path
            mgr = mgr_ref()
            if not mgr:
                return
            loop = mgr._loop
            if loop and not loop.is_closed():
                try:
                    fut = asyncio.run_coroutine_threadsafe(mgr.stop_all(), loop)
                    try:
                        fut.result(timeout=2.5)
                        return
                    except Exception:
                        pass
                except Exception:
                    pass
            # Fallback sync kill
            try:
                mgr.shutdown_sync()
            except Exception:
                pass

        try:
            atexit.register(_cleanup_instance)
            self._atexit_registered = True
        except Exception:
            pass

    async def _prune_loop(self) -> None:
        """Periodic loop that prunes expired or dead servers."""
        try:
            while True:
                await asyncio.sleep(5.0)
                await self._prune_once()
        except asyncio.CancelledError:  # pragma: no cover
            return

    async def _prune_once(self) -> None:
        """Terminate servers that exceeded TTL or already exited.

        Removes instances from registry, cancels log tasks, and unregisters PIDs.
        """
        now = time.time()
        expired: list[str] = []
        async with self._lock:
            for key, inst in list(self._servers.items()):
                if inst.process.returncode is not None:
                    expired.append(key)
                    continue
                if now - inst.last_used > self._ttl_seconds:
                    try:
                        inst.process.terminate()
                    except Exception:
                        pass
                    expired.append(key)
            # Cleanup expired instances (cancel log tasks, ensure process exit)
            for key in expired:
                inst = self._servers.pop(key, None)
                if not inst:
                    continue
                try:
                    if inst.stdout_task:
                        inst.stdout_task.cancel()
                    if inst.stderr_task:
                        inst.stderr_task.cancel()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(inst.process.wait(), timeout=1.0)
                except Exception:
                    try:
                        inst.process.kill()
                    except Exception:
                        pass
                if inst.process.pid is not None:
                    _unregister_pid(inst.process.pid)

    async def touch(self, model: str) -> None:
        """Update the last-used timestamp for a given model's server, if present.

        Args:
            model: Model spec key to touch.
        """
        async with self._lock:
            inst = self._servers.get(model)
            if inst:
                inst.last_used = time.time()

    async def stop_all(self) -> None:
        """Terminate all running servers and cancel background tasks."""
        async with self._lock:
            items = list(self._servers.items())
            self._servers.clear()
        # Terminate outside lock and wait for exit
        for key, inst in items:
            try:
                if inst.process.returncode is None:
                    inst.process.terminate()
            except Exception:
                pass
            try:
                if inst.stdout_task:
                    inst.stdout_task.cancel()
                if inst.stderr_task:
                    inst.stderr_task.cancel()
            except Exception:
                pass
            try:
                await asyncio.wait_for(inst.process.wait(), timeout=2.0)
            except Exception:
                try:
                    inst.process.kill()
                except Exception:
                    pass
            if inst.process.pid is not None:
                _unregister_pid(inst.process.pid)
        if self._pruner_task:
            try:
                self._pruner_task.cancel()
            except Exception:
                pass

    def shutdown_sync(self) -> None:
        """Best-effort synchronous shutdown for interpreter exit.

        Sends SIGTERM then SIGKILL to tracked processes without awaiting their exit.
        Intended for use from atexit when an event loop may be unavailable.

        Notes:
            Normal code paths should prefer the async `stop_all`.
        """
        try:
            procs = []
            for inst in list(self._servers.values()):
                if inst.process.pid is not None:
                    procs.append(inst.process.pid)
            for pid in procs:
                _kill_pid(
                    pid, signal.SIGTERM if hasattr(signal, "SIGTERM") else signal.SIGINT
                )
            try:
                time.sleep(0.3)
            except Exception:
                pass
            for pid in procs:
                _kill_pid(
                    pid,
                    signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM,
                )
                _unregister_pid(pid)
        except Exception:
            pass
