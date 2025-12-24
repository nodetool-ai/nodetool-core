"""
JavaScript Docker Runner (raw stdout/stderr)
===========================================

Executes user-supplied JavaScript in Node.js and streams raw stdout and stderr
lines. No wrapper or serialization.
"""

from __future__ import annotations

from typing import Any

from nodetool.code_runners.runtime_base import StreamRunnerBase
from nodetool.workflows.processing_context import ProcessingContext


class JavaScriptDockerRunner(StreamRunnerBase):
    """Execute JavaScript code inside Node.js Docker and stream results."""

    def __init__(
        self,
        image: str = "node:20-alpine",
        mem_limit: str = "256m",
        nano_cpus: int = 1_000_000_000,
        timeout_seconds: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, timeout_seconds=timeout_seconds, **kwargs)
        self.image = image
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus

    def build_container_command(
        self,
        user_code: str,
        env_locals: dict[str, Any],
    ) -> list[str]:
        user_code_with_args = ""
        for key, value in env_locals.items():
            user_code_with_args += f"const {key} = {value};\n"
        user_code_with_args += user_code
        return ["node", "-e", user_code_with_args]


if __name__ == "__main__":
    # Lightweight smoke test: raw stdout/stderr

    class _SmokeNode:
        def __init__(self) -> None:
            self.id = "smoke-node-js"
            self.outputs: list[tuple[str, Any]] = []

        def add_output(self, slot: str, typ: Any | None = None) -> None:
            self.outputs.append((slot, typ))

    async def _smoke() -> None:
        JavaScriptDockerRunner()
        ctx = ProcessingContext()
        node = _SmokeNode()

        user_code = "console.log('stdout line 0');\nconsole.error('stderr line 1');\n"

        print("[smoke-js] starting...")
        try:
            async for slot, value in JavaScriptDockerRunner().stream(
                user_code=user_code,
                env_locals={"foo": "bar"},
                context=ctx,  # type: ignore[arg-type]
                node=node,  # type: ignore[arg-type]
            ):
                print(f"[stream-js] {slot}: {value}")
        except Exception as exc:
            print(f"[smoke-js] failed: {exc}")
        else:
            print("[smoke-js] complete.")

    import asyncio

    asyncio.run(_smoke())
