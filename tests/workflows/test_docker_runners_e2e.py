from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nodetool.code_runners.bash_runner import BashDockerRunner
from nodetool.code_runners.javascript_runner import JavaScriptDockerRunner
from nodetool.code_runners.python_runner import PythonDockerRunner
from nodetool.code_runners.ruby_runner import RubyDockerRunner
from nodetool.workflows.processing_context import ProcessingContext


def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()  # type: ignore[attr-defined]
        client.ping()
        return True
    except Exception:
        return False


def _image_present(image: str) -> bool:
    try:
        import docker

        client = docker.from_env()
        client.images.get(image)
        return True
    except Exception:
        return False


class _Node:
    def __init__(self, id_: str) -> None:
        self.id = id_


@pytest.mark.asyncio
async def test_python_runner_e2e_with_args(tmp_path: Any) -> None:
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    runner = PythonDockerRunner()
    if not _image_present(runner.docker_image()):
        pytest.skip("Required Docker image not present; skipping to avoid long pulls")
    ctx = ProcessingContext()
    node = _Node("py")
    code = "import sys\nprint('PY_OK')\nprint('ERR', file=sys.stderr)\n"

    async def _collect() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        async for slot, value in runner.stream(
            user_code=code,
            env_locals={},
            context=ctx,  # type: ignore[arg-type]
            node=node,  # type: ignore[arg-type]
        ):
            out.append((slot, value.strip()))
        return out

    out = await asyncio.wait_for(_collect(), timeout=20)

    assert ("stdout", "PY_OK") in out
    assert ("stderr", "ERR") in out
    # Interface updated: args/env not passed; only verify basic stdout/stderr


@pytest.mark.asyncio
async def test_node_runner_e2e_with_args(tmp_path: Any) -> None:
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    runner = JavaScriptDockerRunner()
    if not _image_present(runner.docker_image()):
        pytest.skip("Required Docker image not present; skipping to avoid long pulls")
    ctx = ProcessingContext()
    node = _Node("js")
    code = "console.log('JS_OK');\nconsole.error('ERR');\n"

    async def _collect() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        async for slot, value in runner.stream(
            user_code=code,
            env_locals={},
            context=ctx,  # type: ignore[arg-type]
            node=node,  # type: ignore[arg-type]
        ):
            out.append((slot, value.strip()))
        return out

    out = await asyncio.wait_for(_collect(), timeout=20)

    assert ("stdout", "JS_OK") in out
    assert ("stderr", "ERR") in out
    # Interface updated: args/env not passed; only verify basic stdout/stderr


@pytest.mark.asyncio
async def test_bash_runner_e2e_with_args(tmp_path: Any) -> None:
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    runner = BashDockerRunner()
    if not _image_present(runner.docker_image()):
        pytest.skip("Required Docker image not present; skipping to avoid long pulls")
    ctx = ProcessingContext()
    node = _Node("sh")
    code = "echo START; echo OUT 1; echo ERR 1 1>&2; sleep 1; echo OUT 2; echo ERR 2 1>&2; echo DONE"

    async def _collect() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        async for slot, value in runner.stream(
            user_code=code,
            env_locals={},
            context=ctx,  # type: ignore[arg-type]
            node=node,  # type: ignore[arg-type]
        ):
            out.append((slot, value.strip()))
        return out

    out = await asyncio.wait_for(_collect(), timeout=20)
    print(out)

    assert ("stdout", "START") in out
    assert ("stdout", "OUT 1") in out
    assert ("stderr", "ERR 1") in out
    assert ("stdout", "OUT 2") in out
    assert ("stderr", "ERR 2") in out
    assert ("stdout", "DONE") in out
    # Interface updated: args/env not passed; only verify basic stdout/stderr


@pytest.mark.asyncio
async def test_ruby_runner_e2e_with_args(tmp_path: Any) -> None:
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    runner = RubyDockerRunner()
    if not _image_present(runner.docker_image()):
        pytest.skip("Required Docker image not present; skipping to avoid long pulls")
    ctx = ProcessingContext()
    node = _Node("rb")
    code = "puts 'RB_OK'\nSTDERR.puts 'ERR'\n"

    async def _collect() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        async for slot, value in runner.stream(
            user_code=code,
            env_locals={},
            context=ctx,  # type: ignore[arg-type]
            node=node,  # type: ignore[arg-type]
        ):
            out.append((slot, value.strip()))
        return out

    out = await asyncio.wait_for(_collect(), timeout=20)

    assert ("stdout", "RB_OK") in out
    assert ("stderr", "ERR") in out
    # Interface updated: args/env not passed; only verify basic stdout/stderr
