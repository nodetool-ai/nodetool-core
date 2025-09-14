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
import shlex
import signal
import socket
import time
import atexit
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import weakref

import httpx

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _is_windows_abs_path(path: str) -> bool:
    """Detect Windows absolute paths (e.g., "C:\\...") or UNC paths.

    This helps disambiguate from HF repo specs like "org/repo:file.gguf"
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
    """Heuristically determine if the model string is a filesystem path.

    Rules:
    - POSIX absolute or relative path prefixes: "/", "./", "../"
    - Windows absolute or UNC paths
    - A bare filename ending with ".gguf" that does not contain a colon
      (to avoid misclassifying HF specs like "org/repo:file.gguf")
    """
    if model.startswith("/") or model.startswith("./") or model.startswith("../"):
        return True
    if _is_windows_abs_path(model):
        return True
    if model.lower().endswith(".gguf") and ":" not in model:
        return True
    return False


def _hf_cache_dir() -> str:
    """Return the Hugging Face hub cache directory.

    Respects HUGGINGFACE_HUB_CACHE or HF_HOME; otherwise defaults to
    ~/.cache/huggingface/hub
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
    """Attempt to resolve a repo file to a local path in the HF cache.

    Strategy:
    1) Prefer huggingface_hub.hf_hub_download(local_files_only=True) when available
    2) Fallback to scanning ~/.cache/huggingface/hub snapshots structure
    """
    # 1) Try huggingface_hub API with local-only to avoid implicit downloads
    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        token = os.environ.get("HF_TOKEN") or None
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                local_files_only=True,
            )
            if os.path.exists(path):
                log.debug(
                    f"Resolved HF file locally via huggingface_hub: {repo_id}:{filename} -> {path}"
                )
                return path
        except Exception:
            pass
    except Exception:
        # huggingface_hub not available; fallback to cache scanning
        pass

    # 2) Scan hub snapshots for the file
    try:
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
    except Exception:
        return None

    return None


def _parse_model_args(model: str) -> Tuple[list[str], str]:
    """Return (args, alias) for llama-server based on the model spec.

    alias is used to set --alias so the OAI client can pass a stable model name.
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


if TYPE_CHECKING:  # pragma: no cover - typing hints only
    pass


@dataclasses.dataclass
class _RunningServer:
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
    try:
        _GLOBAL_PIDS.add(pid)
    except Exception:
        pass


def _unregister_pid(pid: int) -> None:
    try:
        _GLOBAL_PIDS.discard(pid)
    except Exception:
        pass


def _kill_pid(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return
    except Exception:
        # Best-effort; ignore
        pass


def _atexit_kill_all() -> None:  # pragma: no cover - runs at interpreter shutdown
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
    def __init__(self, ttl_seconds: Optional[int] = None):
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

    async def ensure_server(self, model: str) -> str:
        """Ensure a llama-server is running for the given model spec and return base URL.

        The server is started on-demand and kept alive until the inactivity TTL elapses.
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
                """Return a safely loggable command string with sensitive values redacted."""
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
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            if proc.pid is not None:
                _register_pid(proc.pid)

            base_url = f"http://{self._host}:{port}"

            # Background async readers to forward server logs to our logger
            async def _reader(stream: asyncio.StreamReader, which: str) -> None:
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
            if proc.stdout is not None:
                t_out = asyncio.create_task(_reader(proc.stdout, "stdout"))
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
        if self._pruner_task is None or self._pruner_task.done():
            self._pruner_task = asyncio.create_task(self._prune_loop())

    def _install_signal_handlers(self) -> None:
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
        try:
            while True:
                await asyncio.sleep(5.0)
                await self._prune_once()
        except asyncio.CancelledError:  # pragma: no cover
            return

    async def _prune_once(self) -> None:
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
        async with self._lock:
            inst = self._servers.get(model)
            if inst:
                inst.last_used = time.time()

    async def stop_all(self) -> None:
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
        """Best-effort sync shutdown for interpreter exit.

        Sends SIGTERM then SIGKILL without awaiting. This is used by atexit as a
        fallback in case the event loop is unavailable. Normal code paths should
        call the async stop_all().
        """
        try:
            procs = []
            for inst in list(self._servers.values()):
                if inst.process.pid is not None:
                    procs.append(inst.process.pid)
            for pid in procs:
                _kill_pid(pid, signal.SIGTERM if hasattr(signal, "SIGTERM") else signal.SIGINT)
            try:
                time.sleep(0.3)
            except Exception:
                pass
            for pid in procs:
                _kill_pid(pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
                _unregister_pid(pid)
        except Exception:
            pass
