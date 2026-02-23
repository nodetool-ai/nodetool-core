import pytest
import json
import shutil
from nodetool.code_runners.javascript_runner import JavaScriptDockerRunner
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext

NODE_AVAILABLE = shutil.which("node") is not None

@pytest.fixture
def runner():
    return JavaScriptDockerRunner(mode="subprocess")

def test_build_container_command():
    runner = JavaScriptDockerRunner()
    user_code = "console.log('hello');"
    env_locals = {"foo": "bar", "num": 123}

    cmd = runner.build_container_command(user_code, env_locals)

    assert cmd[0] == "node"
    assert cmd[1] == "-e"
    # Order of dict iteration is preserved in recent python, but let's check containment
    assert 'const foo = "bar";' in cmd[2]
    assert 'const num = 123;' in cmd[2]
    assert user_code in cmd[2]

@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js not found")
@pytest.mark.asyncio
async def test_javascript_runner_console_log(runner):
    ctx = ProcessingContext()
    node = BaseNode("test-node")

    outputs = []
    async for slot, value in runner.stream(
        user_code="console.log('hello');",
        env_locals={},
        context=ctx,
        node=node
    ):
        if slot == "stdout":
            outputs.append(value.strip())

    assert "hello" in outputs

@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js not found")
@pytest.mark.asyncio
async def test_javascript_runner_env_locals(runner):
    ctx = ProcessingContext()
    node = BaseNode("test-node")

    outputs = []
    async for slot, value in runner.stream(
        user_code="console.log(foo);",
        env_locals={"foo": "bar"},
        context=ctx,
        node=node
    ):
        if slot == "stdout":
            outputs.append(value.strip())

    assert "bar" in outputs

@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js not found")
@pytest.mark.asyncio
async def test_javascript_runner_complex_locals(runner):
    ctx = ProcessingContext()
    node = BaseNode("test-node")

    complex_data = {"a": [1, 2], "b": {"c": "d"}}
    outputs = []
    async for slot, value in runner.stream(
        user_code="console.log(JSON.stringify(data));",
        env_locals={"data": complex_data},
        context=ctx,
        node=node
    ):
        if slot == "stdout":
            outputs.append(value.strip())

    # Check if we got any output
    assert outputs, "No stdout received"

    # Parse the JSON output back to a python object for comparison
    # The output might be split across multiple chunks if very long, but for this small example it should be one line.
    # However, runner streams lines, so outputs[0] should be the full line.
    result = json.loads(outputs[0])
    assert result == complex_data

@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js not found")
@pytest.mark.asyncio
async def test_javascript_runner_stderr(runner):
    ctx = ProcessingContext()
    node = BaseNode("test-node")

    outputs = []
    async for slot, value in runner.stream(
        user_code="console.error('oops');",
        env_locals={},
        context=ctx,
        node=node
    ):
        if slot == "stderr":
            outputs.append(value.strip())

    assert "oops" in outputs
