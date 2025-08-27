"""
Python Docker Runner (raw stdout/stderr)
=======================================

Executes user-supplied Python code inside Docker and streams raw stdout and
stderr lines. No wrapping or serialization is added; the code runs as-is with
`python -c`.
"""

from typing import Any, AsyncGenerator

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.code_runners.runtime_base import StreamRunnerBase


class PythonDockerRunner(StreamRunnerBase):
    """Execute Python code inside Docker and stream raw stdout/stderr."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        mem_limit: str = "256m",
        nano_cpus: int = 1_000_000_000,
        timeout_seconds: int = 10,
    ):
        super().__init__(timeout_seconds=timeout_seconds)
        self.image = image
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus

    # no wrapper or serialization

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

    # Docker hooks for the base class
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
        return ["python", "-c", user_code_with_args]


if __name__ == "__main__":
    # Lightweight smoke test
    import asyncio

    class _SmokeNode:
        def __init__(self) -> None:
            self.id = "smoke-node"

    async def _smoke() -> None:
        ctx = ProcessingContext()
        node = _SmokeNode()
        user_code = (
            "import sys\n" "print('hello stdout')\n" "print('oops', file=sys.stderr)\n"
        )
        async for slot, value in PythonDockerRunner().stream(
            user_code=user_code,
            env_locals={"foo": "bar"},
            context=ctx,  # type: ignore[arg-type]
            node=node,  # type: ignore[arg-type]
        ):
            print(f"[stream] {slot}: {value}")

    asyncio.run(_smoke())
