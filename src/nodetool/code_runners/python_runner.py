"""
Python Docker Runner (raw stdout/stderr)
=======================================

Executes user-supplied Python code inside Docker and streams raw stdout and
stderr lines. No wrapping or serialization is added; the code runs as-is with
`python -c`.
"""

from typing import Any

from nodetool.code_runners.runtime_base import StreamRunnerBase
from nodetool.workflows.processing_context import ProcessingContext


class PythonDockerRunner(StreamRunnerBase):
    """Execute Python code inside Docker and stream raw stdout/stderr."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        **kwargs,
    ):
        super().__init__(
            image=image,
            **kwargs,
        )

    def build_container_command(self, user_code: str, env_locals: dict[str, Any]) -> list[str]:
        user_code_with_args = ""
        for key, value in env_locals.items():
            user_code_with_args += f"{key}={repr(value)}\n"
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
            "import sys\n"
            "print('hello stdout')\n"
            "print('oops', file=sys.stderr)\n"
            "l1 = sys.stdin.readline().strip()\n"
            "l2 = sys.stdin.readline().strip()\n"
            "print(f'stdin1={l1}')\n"
            "print(f'stdin2={l2}')\n"
        )

        async def _stdin_gen():
            yield "first line"
            await asyncio.sleep(0.05)
            yield "second line"

        async for slot, value in PythonDockerRunner().stream(
            user_code=user_code,
            env_locals={"foo": 123},
            context=ctx,  # type: ignore[arg-type]
            node=node,  # type: ignore[arg-type]
            stdin_stream=_stdin_gen(),
        ):
            print(f"[stream] {slot}: {value}")

    asyncio.run(_smoke())
