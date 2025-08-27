from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

import pytest

from nodetool.code_runners.runtime_base import StreamRunnerBase


class _FakeRunner(StreamRunnerBase):
    def __init__(
        self, chunks: list[tuple[str | None, str | bytes]], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._chunks = chunks

    def docker_image(self) -> str:  # pragma: no cover - not used in test
        return "fake:image"

    def build_container_command(
        self, user_code: str, env_locals: dict[str, Any], args: list[str]
    ) -> list[str]:  # pragma: no cover - not used
        return ["/bin/true"]

    # Override sync runner to bypass docker and feed our chunks
    def _docker_sync_run(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        user_code: str,
        env_locals: dict[str, Any],
        context: Any,
        node: Any,
        args: list[str] | None,
        allow_dynamic_outputs: bool,
    ) -> None:
        for stream_name, data in self._chunks:
            if stream_name == "stdout":
                for part in data if isinstance(data, bytes) else data.encode("utf-8"):
                    # accumulate bytes into lines the same way base does (simulate demux lines)
                    pass
        # Simpler: push lines directly like base would after demux
        for stream_name, data in self._chunks:
            text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
            for line in text.split("\n"):
                if line == "":
                    continue
                asyncio.run_coroutine_threadsafe(
                    queue.put(
                        {
                            "type": "yield",
                            "slot": stream_name or "stdout",
                            "value": line,
                        }
                    ),
                    loop,
                )
        asyncio.run_coroutine_threadsafe(queue.put({"type": "final", "ok": True}), loop)


@pytest.mark.asyncio
async def test_base_streams_stdout_and_stderr_lines() -> None:
    chunks = [
        ("stdout", "hello\nworld\n"),
        ("stderr", "oops\nwarn\n"),
        ("stdout", "tail"),
    ]
    runner = _FakeRunner(chunks)

    out: list[tuple[str, str]] = []
    async for slot, value in runner.stream(
        user_code="",
        env_locals={},
        context=type(
            "C", (), {"workspace_dir": "/tmp", "post_message": lambda *a, **k: None}
        )(),
        node=type("N", (), {"id": "n"})(),
    ):
        out.append((slot, value))

    assert ("stdout", "hello") in out
    assert ("stdout", "world") in out
    assert ("stderr", "oops") in out
    assert ("stderr", "warn") in out
    assert ("stdout", "tail") in out


@pytest.mark.asyncio
async def test_env_stringification_in_base() -> None:
    runner = _FakeRunner([])
    env = runner.build_container_environment(
        {"A": 1, "B": True, "C": None, "D": "x"}, []
    )
    # NT_ARGS_JSON and NT_ARGC are also present; only assert the keys from locals
    assert env["A"] == "1"
    assert env["B"] == "True"
    assert env["C"] == "None"
    assert env["D"] == "x"
