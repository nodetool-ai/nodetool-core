"""
Bash Docker Runner (raw stdout/stderr)
=====================================

Executes user-supplied Bash script inside Docker and streams raw stdout and
stderr lines. No wrapper or serialization.
"""

from __future__ import annotations

from typing import Any

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.code_runners.runtime_base import StreamRunnerBase


class BashDockerRunner(StreamRunnerBase):
    """Execute Bash script inside Docker and stream results."""

    def __init__(
        self,
        image: str = "bash:5.2",
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            image=image,
            **kwargs,
        )

    def build_container_command(
        self,
        user_code: str,
        env_locals: dict[str, Any],
    ) -> list[str]:
        user_code_with_args = "set -e\n"
        for key, value in env_locals.items():
            user_code_with_args += f"{key}={repr(value)}\n"
        user_code_with_args += user_code
        return ["bash", "-lc", user_code_with_args]


if __name__ == "__main__":
    # Lightweight smoke test: raw stdout/stderr with input streaming
    import asyncio

    class _SmokeNode:
        def __init__(self) -> None:
            self.id = "smoke-node-bash"

    async def _smoke() -> None:
        ctx = ProcessingContext()
        node = _SmokeNode()
        # Read lines from stdin and echo to stdout and stderr to validate demux
        user_code = (
            "echo START; "
            'while IFS= read -r line; do echo OUT: "$line"; echo ERR: "$line" 1>&2; done; '
            "echo DONE"
        )

        async def stdin_gen():
            # Provide a few lines of input to verify stdin piping
            for line in ["line-1", "line-2", "line-3"]:
                yield line
                await asyncio.sleep(0.05)

        async for slot, value in BashDockerRunner().stream(
            user_code=user_code,
            env_locals={"FOO": "bar"},
            context=ctx,  # type: ignore[arg-type]
            node=node,  # type: ignore[arg-type]
            stdin_stream=stdin_gen(),
        ):
            print(f"[stream-bash] {slot}: {value}")

    asyncio.run(_smoke())
