import asyncio
import logging
import os
import shlex
import socket
import subprocess
import threading
from typing import Any, AsyncGenerator, AsyncIterator

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress, Notification, LogUpdate
from nodetool.code_runners.docker_ws import DockerHijackMultiplexDemuxer
from nodetool.config.logging_config import get_logger


log = get_logger(__name__)


class StreamRunnerBase:
    """Base class for streaming code runners with explicit execution modes.

    Supports two mutually exclusive modes:
    - "docker": run inside a Docker container
    - "subprocess": run as a local subprocess on the host

    The public entrypoint is `stream`, which yields `(slot, value)` tuples
    where `slot` is either ``"stdout"`` or ``"stderr"`` and ``value`` is a
    newline-terminated string. End-of-stream is signaled via a final message
    and the generator completes.
    """

    def __init__(
        self,
        timeout_seconds: int = 10,
        image: str = "bash:5.2",
        mem_limit: str = "256m",
        nano_cpus: int = 1_000_000_000,
        network_disabled: bool = True,
        ipc_mode: str | None = "host",
        mode: str = "docker",
    ) -> None:
        """Initialize the stream runner.

        Args:
            timeout_seconds: Max time in seconds before the container is force
                removed. Set to a positive value to enable a watchdog timer.
            image: Default Docker image to use for execution.
            mem_limit: Docker memory limit (e.g., ``"256m"``, ``"1g"``).
            nano_cpus: CPU quota in Docker nano-CPUs (1e9 = 1 CPU).
        """
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus
        self.network_disabled = network_disabled
        self.ipc_mode = ipc_mode
        # Explicit execution mode: "docker" or "subprocess"
        self.mode = mode
        # Runtime / lifecycle tracking for cooperative shutdown
        self._active_container_id: str | None = None
        self._active_sock: Any | None = None
        self._stopped: bool = False
        self._lock = threading.Lock()

    # ---- Public API ----
    async def stream(
        self,
        user_code: str,
        env_locals: dict[str, Any],
        context: ProcessingContext,
        node: BaseNode,
        stdin_stream: AsyncIterator[str] | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """Run code inside Docker and stream output lines.

        This method sets up a worker thread to perform the blocking Docker
        operations and uses a queue to forward streaming messages back to the
        asyncio loop.

        Args:
            user_code: Source code or command string to execute inside the
                container. Interpretation depends on ``build_container_command``.
            env_locals: Mapping of local variables or parameters exposed to the
                container. Subclasses may decide how these are used.
            context: Processing context for posting progress updates.
            node: Workflow node initiating this run.
            stdin_stream: Optional async iterator of text chunks to forward to
                the container stdin. Chunks are encoded as UTF-8 and written in
                order; EOF is signaled by shutting down the write side.

        Yields:
            Tuples of ``(slot, value)`` where ``slot`` is ``"stdout"`` or
            ``"stderr"`` and ``value`` is a newline-terminated string.

        Raises:
            RuntimeError: If container execution fails.
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        env = {}

        log.debug(
            "stream() start: code=%s timeout=%s",
            user_code,
            self.timeout_seconds,
        )

        target = self._docker_run if (self.mode == "docker") else self._local_run
        worker = threading.Thread(
            target=target,
            kwargs={
                "queue": queue,
                "loop": loop,
                "user_code": user_code,
                "env": env,
                "env_locals": env_locals,
                "context": context,
                "node": node,
                "stdin_stream": stdin_stream,
            },
            daemon=True,
        )
        worker.start()

        while True:
            msg = await queue.get()
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "yield":
                slot = msg.get("slot", "stdout")
                value = msg.get("value")
                yield slot, value
            elif msg.get("type") == "final":
                log.debug("final received: ok=%s", msg.get("ok"))
                if not msg.get("ok", False):
                    raise RuntimeError(
                        f"Execution error: {msg.get('error', 'Unknown error')}"
                    )
                break

    # ---- Public stoppable lifecycle API ----
    def stop(self) -> None:
        """Stop any active container and close the hijacked socket.

        Safe to call multiple times.

        Returns:
            None
        """
        with self._lock:
            self._stopped = True
            sock = self._active_sock
            container_id = self._active_container_id
        # Close socket to unblock demux loop quickly
        try:
            if sock is not None and getattr(sock, "_sock", None) is not None:
                try:
                    sock._sock.close()
                except Exception:
                    pass
        except Exception:
            pass
        # Force remove container
        if container_id:
            try:
                client = self._get_docker_client()
                try:
                    c = client.containers.get(container_id)
                    c.remove(force=True)
                except Exception:
                    pass
            except Exception:
                pass

    def build_container_command(
        self, user_code: str, env_locals: dict[str, Any]
    ) -> list[str]:
        """Return the command list to run inside the container.

        Args:
            user_code: Code or command string provided by the caller.
            env_locals: Mapping of local variables/parameters.

        Returns:
            A list of arguments forming the container process command, e.g.
            ``["bash", "-lc", user_code]``.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_container_environment(
        self,
        env: dict[str, Any],
    ) -> dict[str, str]:
        """Build the environment dict for Docker.

        Converts values to strings; unconvertible values are set to an empty
        string. Subclasses may override to customize behavior.

        Args:
            env: Mapping of environment key-value pairs.

        Returns:
            A string-to-string dictionary suitable for Docker environment.
        """
        out: dict[str, str] = {}
        for k, v in (env or {}).items():
            try:
                out[str(k)] = str(v)
            except Exception:
                out[str(k)] = ""
        return out

    # Common helper expected by tests to query the image name
    def docker_image(self) -> str:
        """Return the Docker image used by this runner."""
        return self.image

    # ---- Docker execution implementation ----
    def _docker_run(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        user_code: str,
        env: dict[str, Any],
        env_locals: dict[str, Any],
        context: ProcessingContext,
        node: BaseNode,
        stdin_stream: AsyncIterator[str] | None = None,
    ) -> None:
        """Blocking Docker workflow executed in a worker thread.

        Sets up and starts the container, streams hijacked stdout/stderr, and
        posts messages back to the asyncio loop via ``queue``.

        Args:
            queue: Thread-safe asyncio queue used to deliver stream events.
            loop: Event loop where queue operations are executed.
            user_code: Code or command string to execute.
            env: Environment mapping for the container.
            env_locals: Additional locals for subclass-specific behavior.
            context: Processing context for progress messages.
            node: Node associated with the execution.
            stdin_stream: Optional async iterator feeding container stdin.
        """
        command_str: str | None = None
        try:
            # Acquire a Docker client; if unavailable, propagate error (no fallback in docker mode)
            client = self._get_docker_client()

            image = self.image
            command = self.build_container_command(user_code, env_locals)
            command_str = self._format_command_str(command)
            environment = self.build_container_environment(env)

            self._log_docker_params(image, command, command_str, environment)

            self._ensure_image(client, image, context, node)

            container = None
            cancel_timer: threading.Timer | None = None
            try:
                container = self._create_container(
                    client, image, command, environment, context, stdin_stream
                )
                sock = self._attach_before_start(container, stdin_stream)
                # Publish active resources for cooperative shutdown
                with self._lock:
                    self._active_container_id = getattr(container, "id", None)
                    self._active_sock = sock
                self._start_container(container, command_str)

                self._start_stdin_feeder(sock, stdin_stream, loop)

                cancel_timer = self._start_timeout_timer(container)

                self._stream_hijacked_output(sock, queue, loop, context, node)
                # Ensure the container process has exited before finalizing
                exit_code = self._wait_for_container_exit(container)

                # Check exit code and fail if non-zero
                if exit_code != 0:
                    raise RuntimeError(
                        f"Container exited with non-zero status: {exit_code}"
                    )

                self._finalize_success(queue, loop)
                log.debug("_docker_run() completed successfully")
            finally:
                self._cleanup_container(container, cancel_timer)
                # Clear active references
                with self._lock:
                    self._active_container_id = None
                    self._active_sock = None
        except Exception as e:
            self._handle_run_exception(e, command_str, queue, loop)

    # ---- Local subprocess fallback ----
    def _local_run(
        self,
        *,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        user_code: str,
        env: dict[str, Any],
        env_locals: dict[str, Any],
        context: ProcessingContext,
        node: BaseNode,
        stdin_stream: AsyncIterator[str] | None,
    ) -> None:
        """Execute the command as a local subprocess and stream stdout/stderr.

        This serves as a graceful degradation when Docker is not available,
        preserving the same streaming semantics as the Docker-backed runner.
        """
        log.debug("_local_run() begin: code=%s", user_code)
        command_vec: list[str] | None = None
        proc: subprocess.Popen[bytes] | None = None
        cancel_timer: threading.Timer | None = None
        try:
            command_vec = self.build_container_command(user_code, env_locals)
            cmd_str = self._format_command_str(command_vec)

            # Prepare environment and working directory
            proc_env = os.environ.copy()
            proc_env.update(self.build_container_environment(env))
            cwd = getattr(context, "workspace_dir", None) or os.getcwd()

            log.debug("starting local subprocess: cmd=%s cwd=%s", cmd_str, cwd)
            proc = subprocess.Popen(
                command_vec or [],
                cwd=cwd,
                env=proc_env,
                # If no stdin stream is provided, explicitly close stdin so
                # commands like `cat` see EOF immediately and do not hang.
                stdin=(
                    subprocess.PIPE if stdin_stream is not None else subprocess.DEVNULL
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # line-buffered
                universal_newlines=False,
            )

            # Start reader threads
            stdout_thread: threading.Thread | None = None
            stderr_thread: threading.Thread | None = None
            if proc.stdout is not None:
                stdout_thread = threading.Thread(
                    target=self._reader_thread,
                    args=(proc.stdout, "stdout", queue, loop, context, node),
                    daemon=True,
                )
                stdout_thread.start()
            if proc.stderr is not None:
                stderr_thread = threading.Thread(
                    target=self._reader_thread,
                    args=(proc.stderr, "stderr", queue, loop, context, node),
                    daemon=True,
                )
                stderr_thread.start()

            # Forward stdin if provided
            if stdin_stream is not None and proc.stdin is not None:

                async def _feed() -> None:
                    assert proc.stdin is not None
                    assert stdin_stream is not None
                    try:
                        async for data in stdin_stream:
                            if not data.endswith("\n"):
                                data = data + "\n"
                            b = data.encode("utf-8")
                            proc.stdin.write(b)
                        proc.stdin.close()
                    except Exception as e:
                        log.debug("local stdin feeder ended: %s", e)

                asyncio.run_coroutine_threadsafe(_feed(), loop)

            # Optional timeout watchdog to terminate long-running local processes
            if self.timeout_seconds and self.timeout_seconds > 0:

                def _force_kill() -> None:
                    try:
                        if proc and proc.poll() is None:
                            log.debug("forcing kill of local subprocess")
                            proc.terminate()
                    except Exception:
                        pass

                cancel_timer = threading.Timer(self.timeout_seconds, _force_kill)
                cancel_timer.daemon = True
                cancel_timer.start()

            rc = proc.wait()
            log.debug("local subprocess exited with code %s", rc)
            # Ensure stdout/stderr reader threads have drained remaining buffered lines
            try:
                if stdout_thread is not None:
                    stdout_thread.join(timeout=0.5)
            except Exception:
                pass
            try:
                if stderr_thread is not None:
                    stderr_thread.join(timeout=0.5)
            except Exception:
                pass
            if rc != 0:
                raise RuntimeError(f"Process exited with code {rc}")

            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "final", "ok": True}), loop
            )
        except Exception as e:
            try:
                log.exception("_local_run() error for cmd=%s: %s", command_vec, e)
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
                            proc.terminate()
                    except Exception:
                        pass
            except Exception:
                pass

    def _reader_thread(
        self,
        pipe,  # IO[bytes]
        slot: str,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        context: ProcessingContext,
        node: BaseNode,
    ) -> None:
        """Read lines from a byte stream and emit them as messages/logs."""
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
            log.debug("reader(%s) ended: %s", slot, e)

    # ---- Helpers (Docker) ----
    def _get_docker_client(self):  # type: ignore[no-untyped-def]
        """Create and validate a Docker client.

        Returns:
            A Docker client obtained from environment configuration.

        Raises:
            RuntimeError: If the Docker daemon is unreachable.
        """
        import docker

        client = docker.from_env()
        try:
            client.ping()
        except Exception:
            raise RuntimeError(
                "Docker daemon is not available. Please start Docker and try again."
            )
        return client

    def _format_command_str(self, command: list[str] | None) -> str | None:
        """Return a shell-quoted string representation of a command list.

        Args:
            command: Command argument vector.

        Returns:
            A single string suitable for logging, or a best-effort string if
            quoting fails, or ``None`` if ``command`` is ``None``.
        """
        try:
            return " ".join(shlex.quote(part) for part in (command or []))
        except Exception:
            return str(command)

    def _log_docker_params(
        self,
        image: str,
        command: list[str] | None,
        command_str: str | None,
        environment: dict[str, str],
    ) -> None:
        """Log sanitized Docker parameters for debugging.

        Args:
            image: Docker image name.
            command: Command argument vector.
            command_str: Shell-quoted command string.
            environment: Environment variables to pass into the container.
        """
        try:
            log.debug(
                "docker params: image=%s mem=%s cpus=%s cmd_list=%s cmd=%s env_keys=%s",
                image,
                self.mem_limit,
                self.nano_cpus,
                command,
                command_str,
                sorted(list(environment.keys()))[:20],
            )
        except Exception:
            pass

    def _ensure_image(
        self,
        client: Any,
        image: str,
        context: ProcessingContext,
        node: BaseNode,
    ) -> None:
        """Ensure the Docker image is available locally, pulling if needed.

        Args:
            client: Docker client.
            image: Image to ensure.
            context: Processing context for progress updates.
            node: Node used for progress attribution.
        """
        try:
            client.images.get(image)
        except Exception:
            log.debug("pulling image: %s", image)
            context.post_message(
                Notification(
                    node_id=node.id,
                    severity="info",
                    content=f"Pulling image: {image}",
                )
            )
            context.post_message(
                LogUpdate(
                    node_id=node.id,
                    node_name=node.get_title(),
                    content=f"Pulling image: {image}",
                    severity="info",
                )
            )
            client.images.pull(image)
            context.post_message(
                Notification(
                    node_id=node.id,
                    severity="info",
                    content=f"Downloaded image: {image}",
                )
            )
            context.post_message(
                LogUpdate(
                    node_id=node.id,
                    node_name=node.get_title(),
                    content=f"Downloaded image: {image}",
                    severity="info",
                )
            )

    def _create_container(
        self,
        client: Any,
        image: str,
        command: list[str] | None,
        environment: dict[str, str],
        context: ProcessingContext,
        stdin_stream: AsyncIterator[str] | None,
    ) -> Any:
        """Create a detached container configured for streaming I/O.

        Args:
            client: Docker client.
            image: Docker image name.
            command: Command to run.
            environment: Environment variables for the container.
            context: Processing context containing the workspace mount.
            stdin_stream: Whether to open stdin based on non-``None``.

        Returns:
            The created container object.
        """
        log.debug("creating container")
        container = client.containers.create(
            image=image,
            command=command,
            network_disabled=self.network_disabled,
            mem_limit=self.mem_limit,
            nano_cpus=self.nano_cpus,
            volumes={
                context.workspace_dir: {
                    "bind": "/workspace",
                    "mode": "rw",
                }
            },
            working_dir="/workspace",
            stdin_open=stdin_stream is not None,
            tty=False,
            detach=True,
            environment=environment,
            ipc_mode=self.ipc_mode,
        )
        log.debug("container created: id=%s", getattr(container, "id", "<no-id>"))
        return container

    def _attach_before_start(self, container: Any, stdin_stream: AsyncIterator[str] | None):  # type: ignore[no-untyped-def]
        """Attach a hijacked socket before starting the container.

        Attaching prior to start ensures no early output is missed.

        Args:
            container: The Docker container object.
            stdin_stream: Optional stdin source to decide whether to attach
                the stdin channel.

        Returns:
            The low-level attachment socket (hijacked HTTP connection).
        """
        # Attach BEFORE starting the container so we never miss the earliest output
        # Use hijacked HTTP socket (non-WS) for robust local docker schemes
        log.debug("attaching hijacked socket to container before start")
        sock = container.attach_socket(
            params={
                "stdout": True,
                "stderr": True,
                "stdin": stdin_stream is not None,
                "stream": True,
                "logs": True,
            },
        )
        return sock

    def _start_container(self, container: Any, command_str: str | None) -> None:  # type: ignore[no-untyped-def]
        """Start the container and log the command if available.

        Args:
            container: Docker container to start.
            command_str: Optional command string for logging.
        """
        container.start()
        log.debug("container started: id=%s", getattr(container, "id", "<no-id>"))
        if command_str is not None:
            log.debug("executing command: %s", command_str)

    def _start_stdin_feeder(
        self,
        sock: Any,
        stdin_stream: AsyncIterator[str] | None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:  # type: ignore[no-untyped-def]
        """Start an async task that forwards text chunks to container stdin.

        The feeder runs on the provided event loop and performs socket I/O in a
        worker thread via ``asyncio.to_thread`` to avoid blocking the loop.

        Args:
            sock: Hijacked Docker socket wrapper with ``_sock`` attribute.
            stdin_stream: Async iterator producing text chunks to write.
            loop: Event loop used to schedule the feeder coroutine.
        """
        # Schedule feeding of stdin on the provided asyncio loop.
        if stdin_stream is None:
            log.debug("no stdin stream provided")
            return

        async def feed_stdin() -> None:
            try:
                bytes_sent = 0

                async for data in stdin_stream:
                    # Ensure each chunk is line-terminated so tools like `cat` emit per-line
                    if not data.endswith("\n"):
                        data = data + "\n"
                    payload = data.encode("utf-8")
                    bytes_sent += len(payload)
                    log.debug("feeding stdin to container: %s", payload)
                    # Avoid blocking the event loop with a socket send
                    sock._sock.sendall(payload)

                sock._sock.shutdown(socket.SHUT_WR)

                log.debug(
                    "fed stdin (hijack): %d bytes and closed stdin",
                    bytes_sent,
                )
            except Exception as e:
                log.debug("stdin feed error: %s", e)

        # Always run on the current loop that owns stream(), per design.
        asyncio.run_coroutine_threadsafe(feed_stdin(), loop)

    def _start_timeout_timer(self, container: Any) -> threading.Timer | None:  # type: ignore[no-untyped-def]
        """Start a watchdog timer that force-removes the container on timeout.

        Args:
            container: Docker container instance.

        Returns:
            The started timer, or ``None`` if no timeout is configured.
        """
        if not (self.timeout_seconds and self.timeout_seconds > 0):
            return None

        def _force_kill() -> None:
            try:
                # If still running, remove forcefully to unblock streams
                log.debug("forcing kill of container")
                container.remove(force=True)
            except Exception:
                pass

        cancel_timer = threading.Timer(self.timeout_seconds, _force_kill)
        cancel_timer.daemon = True
        cancel_timer.start()
        log.debug("timeout timer started: %ss", self.timeout_seconds)
        return cancel_timer

    def _wait_for_container_exit(self, container: Any) -> int:  # type: ignore[no-untyped-def]
        """Block until the container finishes and return its exit code.

        Args:
            container: Docker container instance.

        Returns:
            The container's process exit code (0 for success).
        """
        try:
            res = container.wait()
            # Docker SDK returns {"StatusCode": int, ...}
            status = 0
            if isinstance(res, dict):
                status = int(res.get("StatusCode", 0) or 0)
            else:
                status = int(res or 0)
            log.debug("container exit status: %s", status)
            return status
        except Exception as e:
            # If the container is already gone or API returns an error, log and continue
            log.debug("wait for container exit failed: %s", e)
            return -1

    def _emit_line(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        context: ProcessingContext,
        node: BaseNode,
        slot: str,
        line: str,
    ) -> None:
        """Enqueue a single newline-terminated line to the consumer and post a log update.

        Args:
            queue: Asyncio queue to push the message into.
            loop: Event loop that owns the queue.
            slot: Either ``"stdout"`` or ``"stderr"``.
            line: Line content; a trailing newline is ensured.
        """
        if not line.endswith("\n"):
            line = f"{line}\n"
        log.debug("emit %s: %s", slot, line)
        asyncio.run_coroutine_threadsafe(
            queue.put({"type": "yield", "slot": slot, "value": line}),
            loop,
        )
        try:
            content = line[:-1] if line.endswith("\n") else line
            sev = "info" if slot == "stdout" else "error"
            context.post_message(
                LogUpdate(
                    node_id=node.id,
                    node_name=node.get_title(),
                    content=content,
                    severity=sev,  # type: ignore[arg-type]
                )
            )
        except Exception:
            pass

    def _stream_hijacked_output(
        self,
        sock: Any,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        context: ProcessingContext,
        node: BaseNode,
    ) -> None:
        """Read multiplexed stdout/stderr from the hijacked socket and emit lines.

        Buffers partial lines until a newline is received, then emits complete
        lines to the queue on the provided loop.

        Args:
            sock: Hijacked Docker socket wrapper with ``_sock`` attribute.
            queue: Async queue for delivering messages to the async context.
            loop: Event loop that owns the queue.
        """
        stdout_buf = ""
        stderr_buf = ""
        try:
            demux_recv = DockerHijackMultiplexDemuxer(sock._sock)
            for slot, chunk in demux_recv.iter_messages():
                log.debug("docker: %s %s", str(slot), str(chunk))
                if chunk is None:
                    continue
                if slot == "stdout":
                    text = chunk.decode("utf-8", errors="ignore")
                    if text:
                        stdout_buf += text
                        while "\n" in stdout_buf:
                            line, stdout_buf = stdout_buf.split("\n", 1)
                            self._emit_line(queue, loop, context, node, "stdout", line)
                elif slot == "stderr":
                    text = chunk.decode("utf-8", errors="ignore")
                    if text:
                        stderr_buf += text
                        while "\n" in stderr_buf:
                            line, stderr_buf = stderr_buf.split("\n", 1)
                            self._emit_line(queue, loop, context, node, "stderr", line)
        except Exception as e:
            log.debug("hijack demux loop ended: %s", e)

        # Flush any remaining buffered text as final lines
        if stdout_buf:
            self._emit_line(queue, loop, context, node, "stdout", stdout_buf)
        if stderr_buf:
            self._emit_line(queue, loop, context, node, "stderr", stderr_buf)

    def _finalize_success(
        self, queue: asyncio.Queue[dict[str, Any]], loop: asyncio.AbstractEventLoop
    ) -> None:
        """Signal successful completion to the consumer loop.

        Args:
            queue: Async queue used by the stream.
            loop: Event loop that owns the queue.
        """
        asyncio.run_coroutine_threadsafe(
            queue.put({"type": "final", "ok": True}),
            loop,
        )

    def _cleanup_container(
        self, container: Any | None, cancel_timer: threading.Timer | None
    ) -> None:
        """Best-effort cleanup of timer and container resources.

        Args:
            container: Container to remove.
            cancel_timer: Optional watchdog timer to cancel.
        """
        try:
            if cancel_timer is not None:
                try:
                    cancel_timer.cancel()
                except Exception:
                    pass
            if container is not None:
                log.debug(
                    "removing container: id=%s", getattr(container, "id", "<no-id>")
                )
                container.remove(force=True)
        except Exception:
            pass

    def _handle_run_exception(
        self,
        e: Exception,
        command_str: str | None,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Report an error from the worker thread back to the consumer.

        Args:
            e: The exception that occurred.
            command_str: Optional command string for logging context.
            queue: Async queue for posting the final error.
            loop: Event loop that owns the queue.
        """
        if command_str:
            log.exception(
                "_docker_run() error while running cmd=%s: %s", command_str, e
            )
        else:
            log.exception("_docker_run() error: %s", e)
        asyncio.run_coroutine_threadsafe(
            queue.put({"type": "final", "ok": False, "error": str(e)}), loop
        )
