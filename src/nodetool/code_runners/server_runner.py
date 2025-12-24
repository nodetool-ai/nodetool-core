"""
Server Docker Runner (long-running service with port mapping)
============================================================

Starts a long-running server process inside a Docker container, exposes a
container port on the host (random ephemeral by default), enables networking,
and streams stdout/stderr lines. Yields an initial "endpoint" message with the
reachable host URL (e.g., ws://127.0.0.1:<port>) before regular log streaming.

Intended for cases like running a Playwright WS server or similar HTTP/WS
services that the workflow needs to connect to during execution.
"""

from __future__ import annotations

import asyncio as _asyncio
import socket as _socket
import time as _time
from contextlib import suppress
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from nodetool.workflows.base_node import BaseNode
    from nodetool.workflows.processing_context import ProcessingContext

from .runtime_base import StreamRunnerBase


class ServerDockerRunner(StreamRunnerBase):
    """Run a server process in Docker with exposed port and stream logs.

    Differences from ``StreamRunnerBase``:
      - Networking is enabled by default
      - A container port is published to the host; the runner detects the host
        port and yields a first message on slot "endpoint" with the full URL
      - The rest of stdout/stderr are streamed like the base class

    Notes:
      - The container continues running until stopped or timeout is reached.
      - Consumers can drain the stream concurrently while using the endpoint.
    """

    def __init__(
        self,
        image: str,
        container_port: int,
        scheme: str = "ws",
        host_ip: str = "127.0.0.1",
        timeout_seconds: int = 60,
        mem_limit: str = "256m",
        nano_cpus: int = 1_000_000_000,
        ready_timeout_seconds: int = 15,
        endpoint_path: str = "",
    ) -> None:
        super().__init__(
            image=image,
            timeout_seconds=timeout_seconds,
            mem_limit=mem_limit,
            nano_cpus=nano_cpus,
        )
        self.container_port = int(container_port)
        self.scheme = scheme
        self.host_ip = host_ip
        # Force-enable networking for server containers
        self.network_disabled = False
        self.ready_timeout_seconds = ready_timeout_seconds
        if endpoint_path and not endpoint_path.startswith("/"):
            endpoint_path = "/" + endpoint_path
        self.endpoint_path = endpoint_path

    def build_container_command(self, user_code: str, env_locals: dict[str, Any]) -> list[str]:
        # Default: run via bash -lc to support complex startup commands
        cmd = user_code.strip() or "sleep infinity"
        return ["bash", "-lc", cmd]

    # Override the base worker to publish a port and yield the endpoint
    def _docker_run(
        self,
        queue: _asyncio.Queue[dict[str, Any]],
        loop: _asyncio.AbstractEventLoop,
        user_code: str,
        env: dict[str, Any],
        env_locals: dict[str, Any],
        context: ProcessingContext,
        node: BaseNode,
        allow_dynamic_outputs: bool,
        stdin_stream: AsyncIterator[str] | None = None,
    ) -> None:  # type: ignore[override]
        self._logger.debug(
            "server _docker_run() begin: code=%s image=%s port=%s",
            user_code,
            self.image,
            self.container_port,
        )
        command_str: str | None = None
        try:
            client = self._get_docker_client()

            image = self.image
            command = self.build_container_command(user_code, env_locals)
            command_str = self._format_command_str(command)
            environment = self.build_container_environment(env)

            self._log_docker_params(image, command, command_str, environment)

            self._ensure_image(client, image, context, node)

            container = None
            cancel_timer = None
            try:
                # Create container with published port on host_ip and ephemeral port
                self._logger.debug("creating server container with port mapping")
                ports = {f"{self.container_port}/tcp": (self.host_ip, 0)}
                container = client.containers.create(
                    image=image,
                    command=command,
                    network_disabled=False,
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
                    ports=ports,
                    ipc_mode=self.ipc_mode,
                )
                self._logger.debug(
                    "server container created: id=%s",
                    getattr(container, "id", "<no-id>"),
                )

                sock = self._attach_before_start(container, stdin_stream)
                # Publish active resources for cooperative shutdown
                with self._lock:
                    self._active_container_id = getattr(container, "id", None)
                    self._active_sock = sock
                self._start_container(container, command_str)

                Thread(
                    target=self._stream_hijacked_output,
                    args=(sock, queue, loop, context, node),
                ).start()

                # Resolve host port mapping after start
                host_port = self._wait_for_host_port(container)

                # Wait for server readiness before yielding endpoint
                if not self._wait_for_server_ready(
                    host=self.host_ip,
                    port=host_port,
                    container=container,
                    timeout=self.ready_timeout_seconds,
                ):
                    raise RuntimeError(f"Server did not become ready on {self.host_ip}:{host_port}")

                endpoint = f"{self.scheme}://{self.host_ip}:{host_port}{self.endpoint_path}"

                _asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "yield", "slot": "endpoint", "value": endpoint}),
                    loop,
                )

                # Forward stdin (if any) and start timeout timer
                self._start_stdin_feeder(sock, stdin_stream, loop)
                cancel_timer = self._start_timeout_timer(container)

                with suppress(Exception):
                    self._wait_for_container_exit(container)

                self._finalize_success(queue, loop)
                self._logger.debug("server _docker_run() completed successfully")
            finally:
                try:
                    self._cleanup_container(container, cancel_timer)
                finally:
                    # Clear active references
                    with self._lock:
                        self._active_container_id = None
                        self._active_sock = None
        except Exception as e:
            self._handle_run_exception(e, command_str, queue, loop)

    def _wait_for_host_port(self, container: Any, timeout: float = 20.0) -> int:  # type: ignore[no-untyped-def]
        """Poll Docker for the published host port of the given container.

        Args:
            container: Docker container object.
            timeout: Seconds to wait for port mapping.

        Returns:
            The published host port number.

        Raises:
            RuntimeError: If no port binding appears within the timeout.
        """
        deadline = _time.time() + timeout
        self._logger.debug("waiting for host port %s", self.container_port)
        while _time.time() < deadline:
            try:
                container.reload()
                ports_info = (container.attrs or {}).get("NetworkSettings", {}).get("Ports", {})
                if container.attrs["State"]["Status"] == "exited":
                    raise RuntimeError("Container exited before port was published")
                self._logger.debug("container attrs: %s", container.attrs)
                binds = ports_info.get(f"{self.container_port}/tcp")
                if binds and isinstance(binds, list) and binds[0].get("HostPort"):
                    return int(binds[0]["HostPort"])  # type: ignore[arg-type]
            except Exception:
                pass
            _time.sleep(0.2)
        raise RuntimeError("Failed to resolve published host port for server container")

    def _wait_for_server_ready(
        self,
        host: str,
        port: int,
        container: Any,
        timeout: float = 15.0,
    ) -> bool:  # type: ignore[no-untyped-def]
        """Attempt TCP connections until the server accepts or timeout.

        Args:
            host: Host IP to connect to.
            port: Published host port.
            container: Docker container for status checks.
            timeout: Seconds to wait for readiness.

        Returns:
            True if a TCP connection was established; False otherwise.
        """
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            try:
                with _socket.create_connection((host, int(port)), timeout=1.0):
                    return True
            except Exception:
                try:
                    container.reload()
                    # If container stopped/exited, abort early
                    status = getattr(container, "status", None)
                    if status and status not in ("created", "restarting", "running"):
                        break
                except Exception:
                    pass
                _time.sleep(0.2)
        return False
