"""
Ruby Docker Runner (raw stdout/stderr)
=====================================

Executes user-supplied Ruby code and streams raw stdout and stderr lines. No
wrapper or serialization.
"""

from __future__ import annotations

from typing import Any

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.code_runners.runtime_base import StreamRunnerBase


class RubyDockerRunner(StreamRunnerBase):
    """Execute Ruby code inside Docker and stream results."""

    def __init__(
        self,
        image: str = "ruby:3.3-alpine",
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            image=image,
            **kwargs,
        )

    def build_container_command(
        self, user_code: str, env_locals: dict[str, Any]
    ) -> list[str]:
        user_code_with_args = ""
        for key, value in env_locals.items():
            user_code_with_args += f"{key}={repr(value)}\n"
        user_code_with_args += user_code
        return ["ruby", "-e", user_code_with_args]


if __name__ == "__main__":
    # Lightweight smoke test: verify stdin piping and demux
    class _SmokeNode:
        def __init__(self) -> None:
            self.id = "smoke-node-ruby"
            self.outputs: list[tuple[str, Any]] = []

        def add_output(self, slot: str, typ: Any | None = None) -> None:
            self.outputs.append((slot, typ))

    async def _smoke() -> None:
        RubyDockerRunner()
        ctx = ProcessingContext()
        node = _SmokeNode()

        # Ruby script echoes stdin to stdout, and mirrors to stderr for demux test
        user_code = (
            "STDOUT.sync = true; STDERR.sync = true; "
            "STDOUT.puts 'START'; "
            "ARGF.each_line { |line| STDOUT.puts 'OUT: ' + line.strip; STDERR.puts 'ERR: ' + line.strip }; "
            "STDOUT.puts 'DONE'"
        )

        import asyncio

        async def stdin_gen():
            for line in ["line-1", "line-2", "line-3"]:
                yield line
                await asyncio.sleep(0.05)

        print("[smoke-ruby] starting...")
        try:
            async for slot, value in RubyDockerRunner().stream(
                user_code=user_code,
                env_locals={"foo": "bar"},
                context=ctx,  # type: ignore[arg-type]
                node=node,  # type: ignore[arg-type]
                stdin_stream=stdin_gen(),
            ):
                print(f"[stream-ruby] {slot}: {value}")
        except Exception as exc:
            print(f"[smoke-ruby] failed: {exc}")
        else:
            print("[smoke-ruby] complete.")

    import asyncio

    asyncio.run(_smoke())
