"""
Generic Command Docker Runner (single command via shell)
=======================================================

Runs a single command string inside a Docker container using a shell
(defaults to bash). Streams raw stdout and stderr lines without any
serialization or code wrapping.
"""

from __future__ import annotations

from typing import Any

from nodetool.code_runners.runtime_base import StreamRunnerBase
from nodetool.workflows.processing_context import ProcessingContext


class CommandDockerRunner(StreamRunnerBase):
    """Execute a single shell command inside Docker and stream results.

    By default, uses the `bash:5.2` image and executes with `bash -lc` so
    that shell features (globbing, pipes, &&, env var expansion) are
    available. You can switch to a POSIX shell by setting `shell="sh"`
    and `image="alpine:3"` or similar.
    """

    def __init__(
        self,
        image: str = "bash:5.2",
        *args,
        **kwargs,
    ) -> None:
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
        return user_code.split(" ")


if __name__ == "__main__":
    # Lightweight smoke test: verify stdin piping and stdout demux
    import asyncio

    class _SmokeNode:
        def __init__(self) -> None:
            self.id = "smoke-node-command"

    async def _smoke() -> None:
        ctx = ProcessingContext()
        node = _SmokeNode()
        # Use a simple program that echoes stdin to stdout to validate piping
        user_code = "cat"

        async def stdin_gen():
            # Provide a few lines of input to verify stdin piping
            for line in ["line-1", "line-2", "line-3"]:
                yield line
                await asyncio.sleep(0.05)

        async for slot, value in CommandDockerRunner(mode="subprocess").stream(
            user_code=user_code,
            env_locals={},
            context=ctx,  # type: ignore[arg-type]
            node=node,  # type: ignore[arg-type]
            stdin_stream=stdin_gen(),
        ):
            print(f"[stream-cmd] {slot}: {value}")

    asyncio.run(_smoke())
