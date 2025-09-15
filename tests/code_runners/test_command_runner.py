from __future__ import annotations

import asyncio
import os
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.code_runners.command_runner import CommandDockerRunner


class _Node:
    def __init__(self, id: str) -> None:
        self.id = id


def _runner() -> CommandDockerRunner:
    # Use subprocess mode on Windows where Docker may be unavailable
    if os.name == "nt":
        return CommandDockerRunner(mode="subprocess")
    return CommandDockerRunner()


@pytest.mark.asyncio
async def test_command_runner_seq_streams_lines():
    """seq 3 should yield three stdout lines: 1, 2, 3."""
    ctx = ProcessingContext()
    node = _Node("n-seq")

    out: list[tuple[str, str]] = []
    async for slot, value in _runner().stream(
        user_code="seq 3",
        env_locals={},
        context=ctx,
        node=node,  # type: ignore[arg-type]
    ):
        out.append((slot, str(value).strip()))

    # Extract stdout lines only
    stdout = [v for s, v in out if s == "stdout"]
    assert stdout == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_command_runner_cat_streams_stdin_lines():
    """cat should echo stdin lines as stdout lines in order."""
    ctx = ProcessingContext()
    node = _Node("n-cat")

    async def stdin_gen():
        for line in ["alpha", "beta", "gamma"]:
            yield line
            await asyncio.sleep(0.02)

    out: list[tuple[str, str]] = []
    async for slot, value in _runner().stream(
        user_code="cat",
        env_locals={},
        context=ctx,
        node=node,  # type: ignore[arg-type]
        stdin_stream=stdin_gen(),
    ):
        out.append((slot, str(value).strip()))

    stdout = [v for s, v in out if s == "stdout"]
    assert stdout == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
async def test_command_runner_cat_handles_empty_and_flush():
    """When no stdin is provided, cat should not hang and produce no stdout."""
    ctx = ProcessingContext()
    node = _Node("n-cat-empty")

    out: list[tuple[str, str]] = []
    async for slot, value in _runner().stream(
        user_code="cat",
        env_locals={},
        context=ctx,
        node=node,  # type: ignore[arg-type]
        stdin_stream=None,
    ):
        out.append((slot, str(value).strip()))

    # No stdout lines expected when stdin is not opened
    stdout = [v for s, v in out if s == "stdout"]
    assert stdout == []

