"""
Ruby Docker Runner (raw stdout/stderr)
=====================================

Executes user-supplied Ruby code and streams raw stdout and stderr lines. No
wrapper or serialization.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.code_runners.runtime_base import StreamRunnerBase


class RubyDockerRunner(StreamRunnerBase):
    """Execute Ruby code inside Docker and stream results."""

    def __init__(
        self,
        image: str = "ruby:3.3-alpine",
        mem_limit: str = "256m",
        nano_cpus: int = 1_000_000_000,
        timeout_seconds: int = 10,
    ):
        super().__init__(timeout_seconds=timeout_seconds)
        self.image = image
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus

    async def stream(
        self,
        user_code: str,
        env_locals: dict[str, Any],
        context: ProcessingContext,
        node: BaseNode,
        allow_dynamic_outputs: bool = True,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        async for item in super().stream(
            user_code=user_code,
            env_locals=env_locals,
            context=context,
            node=node,
            allow_dynamic_outputs=allow_dynamic_outputs,
        ):
            yield item

    # Docker hooks
    def docker_image(self) -> str:
        return self.image

    def docker_mem_limit(self) -> str:
        return self.mem_limit

    def docker_nano_cpus(self) -> int:
        return self.nano_cpus

    def build_container_command(
        self, user_code: str, env_locals: dict[str, Any]
    ) -> list[str]:
        user_code_with_args = ""
        for key, value in env_locals.items():
            user_code_with_args += f"{key}={value}\n"
        user_code_with_args += user_code
        return ["ruby", "-e", user_code_with_args]


if __name__ == "__main__":
    # Lightweight smoke test: raw stdout/stderr
    class _SmokeNode:
        def __init__(self) -> None:
            self.id = "smoke-node-ruby"
            self.outputs: list[tuple[str, Any]] = []

        def add_output(self, slot: str, typ: Any | None = None) -> None:
            self.outputs.append((slot, typ))

    async def _smoke() -> None:
        runner = RubyDockerRunner()
        ctx = ProcessingContext()
        node = _SmokeNode()

        user_code = "STDOUT.puts 'stdout line 0'\n" "STDERR.puts 'stderr line 1'\n"

        print("[smoke-ruby] starting...")
        try:
            async for slot, value in RubyDockerRunner().stream(
                user_code=user_code,
                env_locals={"foo": "bar"},
                context=ctx,  # type: ignore[arg-type]
                node=node,  # type: ignore[arg-type]
            ):
                print(f"[stream-ruby] {slot}: {value}")
        except Exception as exc:
            print(f"[smoke-ruby] failed: {exc}")
        else:
            print("[smoke-ruby] complete.")

    import asyncio

    asyncio.run(_smoke())
