from __future__ import annotations

import asyncio as _asyncio
import json as _json
import logging as _logging
import threading as _threading
import time as _time
from typing import Any, AsyncGenerator

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import NodeProgress


class StreamRunnerBase:
    """Abstract base for Docker-backed language runtime stream runners.

    Subclasses provide container configuration. The base class handles Docker
    lifecycle and async streaming of raw stdout and stderr without any
    serialization or code wrapping.
    """

    def __init__(self, timeout_seconds: int = 10) -> None:
        self.timeout_seconds = timeout_seconds
        self._logger = _logging.getLogger(__name__)
        self._logger.setLevel(_logging.DEBUG)

    # ---- Public API ----
    async def stream(
        self,
        user_code: str,
        env_locals: dict[str, Any],
        context: ProcessingContext,
        node: BaseNode,
        allow_dynamic_outputs: bool = True,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        queue: _asyncio.Queue[dict[str, Any]] = _asyncio.Queue()
        loop = _asyncio.get_running_loop()
        env = {}

        self._logger.debug(
            "stream() start: node_id=%s timeout=%s",
            getattr(node, "id", "<unknown>"),
            self.timeout_seconds,
        )

        worker = _threading.Thread(
            target=self._docker_sync_run,
            kwargs={
                "queue": queue,
                "loop": loop,
                "user_code": user_code,
                "env": env,
                "env_locals": env_locals,
                "context": context,
                "node": node,
                "allow_dynamic_outputs": allow_dynamic_outputs,
            },
            daemon=True,
        )
        worker.start()

        while True:
            msg = await queue.get()
            self._logger.debug("stream() received msg: %s", msg.get("type"))
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "yield":
                slot = msg.get("slot", "stdout")
                value = msg.get("value")
                # Avoid logging entire content to reduce noise
                try:
                    preview = str(value)
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                except Exception:
                    preview = "<unrepr>"
                self._logger.debug("yield: slot=%s preview=%s", slot, preview)
                yield slot, value
            elif msg.get("type") == "final":
                self._logger.debug("final received: ok=%s", msg.get("ok"))
                if not msg.get("ok", False):
                    raise RuntimeError(
                        f"Execution error: {msg.get('error', 'Unknown error')}"
                    )
                break

    # ---- Hooks required from subclasses ----
    def docker_image(self) -> str:
        raise NotImplementedError

    def docker_mem_limit(self) -> str:
        return "256m"

    def docker_nano_cpus(self) -> int:
        return 1_000_000_000

    def build_container_command(
        self, user_code: str, env_locals: dict[str, Any]
    ) -> list[str]:
        raise NotImplementedError

    def build_container_environment(
        self,
        env: dict[str, Any],
    ) -> dict[str, str]:
        """Convert provided locals to string environment variables for the container.

        Subclasses may override for custom behavior.
        """
        out: dict[str, str] = {}
        for k, v in (env or {}).items():
            try:
                out[str(k)] = str(v)
            except Exception:
                out[str(k)] = ""
        return out

    # ---- Docker execution implementation ----
    def _docker_sync_run(
        self,
        queue: _asyncio.Queue[dict[str, Any]],
        loop: _asyncio.AbstractEventLoop,
        user_code: str,
        env: dict[str, Any],
        env_locals: dict[str, Any],
        context: ProcessingContext,
        node: BaseNode,
        allow_dynamic_outputs: bool,
    ) -> None:
        self._logger.debug(
            "_docker_sync_run() begin: node_id=%s image=%s",
            getattr(node, "id", "<unknown>"),
            self.docker_image(),
        )
        try:
            import docker

            client = docker.from_env()
            try:
                client.ping()
            except Exception:
                raise RuntimeError(
                    "Docker daemon is not available. Please start Docker and try again."
                )

            image = self.docker_image()
            command = self.build_container_command(user_code, env_locals)
            environment = self.build_container_environment(env)

            # Log sanitized execution parameters
            try:
                self._logger.debug(
                    "docker params: image=%s mem=%s cpus=%s cmd=%s env_keys=%s",
                    image,
                    self.docker_mem_limit(),
                    self.docker_nano_cpus(),
                    command,
                    sorted(list(environment.keys()))[:20],
                )
            except Exception:
                pass

            try:
                client.images.get(image)
            except Exception:
                self._logger.debug("pulling image: %s", image)
                context.post_message(
                    NodeProgress(
                        node_id=node.id,
                        progress=0,
                        total=100,
                        chunk=f"Pulling image: {image}",
                    )
                )
                client.images.pull(image)

            container = None
            cancel_timer: _threading.Timer | None = None
            try:
                self._logger.debug("creating container")
                container = client.containers.create(
                    image=image,
                    command=command,
                    network_disabled=True,
                    mem_limit=self.docker_mem_limit(),
                    nano_cpus=self.docker_nano_cpus(),
                    volumes={
                        context.workspace_dir: {
                            "bind": "/workspace",
                            "mode": "rw",
                        }
                    },
                    working_dir="/workspace",
                    stdin_open=False,
                    tty=False,
                    detach=True,
                    environment=environment,
                )
                self._logger.debug(
                    "container created: id=%s", getattr(container, "id", "<no-id>")
                )
                # Attach BEFORE starting the container so we never miss the earliest output
                # (e.g., very short-lived commands). logs=True ensures we still receive any
                # data emitted prior to attach in edge cases.
                self._logger.debug("attaching to container stream (demux) before start")
                stream = container.attach(
                    stdout=True,
                    stderr=True,
                    stream=True,
                    logs=True,
                    demux=True,
                )

                container.start()
                self._logger.debug(
                    "container started: id=%s", getattr(container, "id", "<no-id>")
                )

                # Safety: force-stop/remove container after timeout to avoid hangs
                if self.timeout_seconds and self.timeout_seconds > 0:

                    def _force_kill() -> None:
                        try:
                            # If still running, remove forcefully to unblock streams
                            container.remove(force=True)
                        except Exception:
                            pass

                    cancel_timer = _threading.Timer(self.timeout_seconds, _force_kill)
                    cancel_timer.daemon = True
                    cancel_timer.start()
                    self._logger.debug(
                        "timeout timer started: %ss", self.timeout_seconds
                    )

                stdout_buf = ""
                stderr_buf = ""

                def _emit_line(slot: str, line: str) -> None:
                    # Ensure each emitted line is newline-terminated for consumers expecting line breaks
                    if not line.endswith("\n"):
                        line = f"{line}\n"
                    self._logger.debug("emit line: slot=%s preview=%s", slot, line)
                    _asyncio.run_coroutine_threadsafe(
                        queue.put({"type": "yield", "slot": slot, "value": line}),
                        loop,
                    )

                for out_chunk, err_chunk in stream:
                    if out_chunk is not None:
                        self._logger.debug(
                            "received stdout chunk: %d bytes", len(out_chunk)
                        )
                    if out_chunk:
                        text = out_chunk.decode("utf-8", errors="ignore")
                        if text:
                            stdout_buf += text
                            while "\n" in stdout_buf:
                                line, stdout_buf = stdout_buf.split("\n", 1)
                                _emit_line("stdout", line)
                    if err_chunk is not None:
                        self._logger.debug(
                            "received stderr chunk: %d bytes", len(err_chunk)
                        )
                    if err_chunk:
                        text = err_chunk.decode("utf-8", errors="ignore")
                        if text:
                            stderr_buf += text
                            while "\n" in stderr_buf:
                                line, stderr_buf = stderr_buf.split("\n", 1)
                                _emit_line("stderr", line)

                # Flush any remaining buffered text as final lines
                if stdout_buf:
                    _emit_line("stdout", stdout_buf)
                if stderr_buf:
                    _emit_line("stderr", stderr_buf)

                _asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "final", "ok": True}),
                    loop,
                )
                self._logger.debug("_docker_sync_run() completed successfully")
            finally:
                try:
                    if cancel_timer is not None:
                        try:
                            cancel_timer.cancel()
                        except Exception:
                            pass
                    if container is not None:
                        self._logger.debug(
                            "removing container: id=%s",
                            getattr(container, "id", "<no-id>"),
                        )
                        container.remove(force=True)
                except Exception:
                    pass
        except Exception as e:
            self._logger.exception("_docker_sync_run() error: %s", e)
            _asyncio.run_coroutine_threadsafe(
                queue.put({"type": "final", "ok": False, "error": str(e)}), loop
            )
