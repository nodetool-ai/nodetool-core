from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PARAM_CASES = [
    ("", "Hello, "),
    (json.dumps({"text": "World!"}), "Hello, World!"),
]


# Mark subprocess/CLI tests to run sequentially to avoid conflicts
pytestmark = pytest.mark.xdist_group(name="subprocess_execution")


def _build_simple_workflow_graph() -> dict[str, object]:
    return {
        "nodes": [
            {
                "id": "input_text",
                "type": "nodetool.input.StringInput",
                "data": {
                    "name": "text",
                    "label": "Input Text",
                    "value": "",
                },
            },
            {
                "id": "format_text",
                "type": "nodetool.text.FormatText",
                "data": {
                    "template": "Hello, {{ text }}",
                    "inputs": {"text": {"node_id": "input_text", "output": "value"}},
                },
            },
            {
                "id": "output_result",
                "type": "nodetool.output.StringOutput",
                "data": {
                    "name": "result",
                    "value": "",
                    "inputs": {"value": {"node_id": "format_text", "output": "value"}},
                },
            },
        ],
        "edges": [
            {
                "id": "edge_input_to_format",
                "source": "input_text",
                "sourceHandle": "output",
                "target": "format_text",
                "targetHandle": "text",
            },
            {
                "id": "edge_format_to_output",
                "source": "format_text",
                "sourceHandle": "output",
                "target": "output_result",
                "targetHandle": "value",
            },
        ],
    }


@pytest.mark.parametrize("stdin_text,expected", PARAM_CASES)
def test_run_workflow_cli_subprocess_file(stdin_text, expected, tmp_path):
    request = {
        "workflow_id": "wf_test",
        "user_id": "user_subprocess",
        "auth_token": "token",
        "graph": _build_simple_workflow_graph(),
        "params": {},
    }

    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    python_executable = sys.executable
    module = "nodetool.workflows.run_workflow_cli"
    cmd = [python_executable, "-m", module, str(request_path)]

    result = subprocess.run(
        cmd,
        input=stdin_text,
        text=True,
        capture_output=True,
        cwd=Path.cwd(),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith("{")]
    assert stdout_lines

    final_line = stdout_lines[-1]
    assert final_line["type"] == "job_update"
    assert final_line["status"] == "completed"
    assert "result" in final_line

    output_values = final_line["result"].get("result")
    assert isinstance(output_values, list)
    assert output_values
    message = output_values[0]

    assert message == expected


@pytest.mark.parametrize("params,expected", PARAM_CASES)
def test_run_workflow_cli_subprocess_stdin(params, expected, tmp_path):
    """Test running workflow CLI with JSON via stdin (no file argument)."""
    request = {
        "workflow_id": "wf_test",
        "user_id": "user_subprocess",
        "auth_token": "token",
        "graph": _build_simple_workflow_graph(),
        "params": json.loads(params) if params else {},
    }

    # Convert request to JSON for stdin
    request_json = json.dumps(request)

    python_executable = sys.executable
    module = "nodetool.workflows.run_workflow_cli"
    cmd = [python_executable, "-m", module]  # No file argument

    result = subprocess.run(
        cmd,
        input=request_json,  # Pass full request via stdin
        text=True,
        capture_output=True,
        cwd=Path.cwd(),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith("{")]
    assert stdout_lines

    final_line = stdout_lines[-1]
    assert final_line["type"] == "job_update"
    assert final_line["status"] == "completed"
    assert "result" in final_line

    output_values = final_line["result"].get("result")
    assert isinstance(output_values, list)
    assert output_values
    message = output_values[0]

    assert message == expected


@pytest.mark.parametrize("params,expected", PARAM_CASES)
def test_unified_run_command_stdin(params, expected, tmp_path):
    """Test the unified 'nodetool run --stdin --jsonl' command."""
    request = {
        "workflow_id": "wf_test",
        "user_id": "user_unified",
        "auth_token": "token",
        "graph": _build_simple_workflow_graph(),
        "params": json.loads(params) if params else {},
    }

    # Convert request to JSON for stdin
    request_json = json.dumps(request)

    # Use the unified 'nodetool run' command
    cmd = ["nodetool", "run", "--stdin", "--jsonl"]

    result = subprocess.run(
        cmd,
        input=request_json,  # Pass full request via stdin
        text=True,
        capture_output=True,
        cwd=Path.cwd(),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith("{")]
    assert stdout_lines

    final_line = stdout_lines[-1]
    assert final_line["type"] == "job_update"
    assert final_line["status"] == "completed"
    assert "result" in final_line

    output_values = final_line["result"].get("result")
    assert isinstance(output_values, list)
    assert output_values
    message = output_values[0]

    assert message == expected


@pytest.mark.parametrize("params,expected", PARAM_CASES)
def test_unified_run_command_file(params, expected, tmp_path):
    """Test the unified 'nodetool run --jsonl' command with file argument."""
    request = {
        "workflow_id": "wf_test",
        "user_id": "user_unified",
        "auth_token": "token",
        "graph": _build_simple_workflow_graph(),
        "params": json.loads(params) if params else {},
    }

    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    # Use the unified 'nodetool run' command with file
    cmd = ["nodetool", "run", str(request_path), "--jsonl"]

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=Path.cwd(),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith("{")]
    assert stdout_lines

    final_line = stdout_lines[-1]
    assert final_line["type"] == "job_update"
    assert final_line["status"] == "completed"
    assert "result" in final_line

    output_values = final_line["result"].get("result")
    assert isinstance(output_values, list)
    assert output_values
    message = output_values[0]

    assert message == expected
