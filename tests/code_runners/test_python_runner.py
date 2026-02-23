from __future__ import annotations

import pytest

from nodetool.code_runners.python_runner import PythonDockerRunner
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


def test_build_container_command_basic():
    """Verify that build_container_command correctly constructs the command."""
    runner = PythonDockerRunner()
    user_code = "print('hello world')"
    env_locals = {}

    command = runner.build_container_command(user_code, env_locals)

    assert command == ["python", "-c", "print('hello world')"]


def test_build_container_command_with_locals():
    """Verify that build_container_command injects locals into the code."""
    runner = PythonDockerRunner()
    user_code = "print(x)"
    env_locals = {"x": 10, "y": "test"}

    command = runner.build_container_command(user_code, env_locals)

    # Check that locals are injected before the user code
    assert command[0] == "python"
    assert command[1] == "-c"

    code_arg = command[2]
    assert "x=10" in code_arg
    assert "y='test'" in code_arg or "y=\"test\"" in code_arg
    assert "print(x)" in code_arg

    # Verify the order: locals definition should come before user code
    assert code_arg.index("x=10") < code_arg.index("print(x)")


@pytest.mark.asyncio
async def test_python_runner_subprocess_stdout():
    """Verify that Python runner executes code and captures stdout in subprocess mode."""
    # Use subprocess mode to avoid Docker dependency in tests
    runner = PythonDockerRunner(mode="subprocess")
    ctx = ProcessingContext()
    node = BaseNode("n-test-stdout")

    user_code = "print('hello from python')"
    env_locals = {}

    out: list[tuple[str, str]] = []
    async for slot, value in runner.stream(
        user_code=user_code,
        env_locals=env_locals,
        context=ctx,
        node=node, # type: ignore[arg-type]
    ):
        out.append((slot, str(value).strip()))

    stdout = [v for s, v in out if s == "stdout"]
    assert "hello from python" in stdout


@pytest.mark.asyncio
async def test_python_runner_subprocess_stderr():
    """Verify that Python runner executes code and captures stderr in subprocess mode."""
    runner = PythonDockerRunner(mode="subprocess")
    ctx = ProcessingContext()
    node = BaseNode("n-test-stderr")

    user_code = "import sys; print('error message', file=sys.stderr)"
    env_locals = {}

    out: list[tuple[str, str]] = []
    async for slot, value in runner.stream(
        user_code=user_code,
        env_locals=env_locals,
        context=ctx,
        node=node, # type: ignore[arg-type]
    ):
        out.append((slot, str(value).strip()))

    stderr = [v for s, v in out if s == "stderr"]
    assert "error message" in stderr


@pytest.mark.asyncio
async def test_python_runner_subprocess_locals():
    """Verify that locals are correctly passed to the subprocess execution."""
    runner = PythonDockerRunner(mode="subprocess")
    ctx = ProcessingContext()
    node = BaseNode("n-test-locals")

    user_code = "print(f'value is {my_var}')"
    env_locals = {"my_var": 42}

    out: list[tuple[str, str]] = []
    async for slot, value in runner.stream(
        user_code=user_code,
        env_locals=env_locals,
        context=ctx,
        node=node, # type: ignore[arg-type]
    ):
        out.append((slot, str(value).strip()))

    stdout = [v for s, v in out if s == "stdout"]
    assert "value is 42" in stdout
