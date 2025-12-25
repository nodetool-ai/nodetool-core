"""
Standalone workflow execution CLI for subprocess isolation.

This module provides a low-level interface for running workflows and streaming
JSONL output. It's designed to be called as a subprocess for isolated execution.

DEPRECATED: This module is now deprecated in favor of the unified 'nodetool run' command.
For all use cases, prefer using:

    nodetool run --stdin --jsonl    # For stdin input with JSONL output
    nodetool run workflow.json --jsonl    # For file input with JSONL output
    nodetool run workflow_id    # For interactive mode (pretty-printed)

This module is kept for backward compatibility but may be removed in future versions.
The unified command provides the same JSONL streaming behavior plus interactive mode.
"""

import argparse
import asyncio
import base64
import json
import sys
from contextlib import suppress
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.types.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow

log = get_logger(__name__)


def _default(obj: Any) -> Any:
    """
    JSON serializer for objects not serializable by default json code.
    - Pydantic models: use model_dump
    - Bytes: base64-encode to string
    """
    try:
        # pydantic v2 model
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            return obj.model_dump()
    except Exception:
        pass

    if isinstance(obj, (bytes, bytearray)):
        return {
            "__type__": "bytes",
            "base64": base64.b64encode(bytes(obj)).decode("utf-8"),
        }

    # Fallback to string
    return str(obj)


def _parse_request_json(value: str) -> dict[str, Any]:
    # Allow passing a file path or inline JSON
    try:
        # Try to open as a file path
        with open(value, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        pass

    # Otherwise treat as JSON string
    return json.loads(value)


def _read_stdin_json_or_none() -> Any:
    if sys.stdin.isatty():
        return None
    data = sys.stdin.read()
    if not data.strip():
        return None
    try:
        return json.loads(data)
    except Exception:
        # If stdin contains non-JSON, ignore
        return None


def _merge_params(base: Any, override: Any) -> Any:
    if base is None:
        return override
    if override is None:
        return base
    if isinstance(base, dict) and isinstance(override, dict):
        result = dict(base)
        result.update(override)
        return result
    # Default to override
    return override


async def _run(req_dict: dict[str, Any]) -> int:
    # Build RunJobRequest, handling Graph if passed as plain dict
    graph_value = req_dict.get("graph")
    if isinstance(graph_value, dict):
        with suppress(Exception):
            req_dict["graph"] = Graph(**graph_value)

    req = RunJobRequest(**req_dict)

    context = ProcessingContext(
        user_id=req.user_id,
        auth_token=req.auth_token,
        workflow_id=req.workflow_id,
        job_id=None,
    )

    try:
        async for msg in run_workflow(
            req,
            context=context,
            use_thread=False,
            send_job_updates=True,
            initialize_graph=True,
            validate_graph=True,
        ):
            line = json.dumps(msg if isinstance(msg, dict) else _default(msg), default=_default)
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        return 0
    except Exception as e:
        err = {"type": "error", "error": str(e)}
        sys.stdout.write(json.dumps(err) + "\n")
        sys.stdout.flush()
        return 1


def main() -> int:
    import warnings
    warnings.warn(
        "run_workflow_cli.main() is deprecated. Use 'nodetool run --stdin --jsonl' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(
        description=(
            "Run a workflow from a RunJobRequest JSON and stream JSONL messages to stdout.\n"
            "- If a positional arg is provided, it can be a JSON string or a path to a JSON file.\n"
            "- If no positional arg, reads the full RunJobRequest JSON from stdin.\n"
            "- If positional arg is provided, stdin may optionally provide JSON params to merge."
        )
    )
    parser.add_argument(
        "request_json",
        nargs="?",  # Make it optional
        help="RunJobRequest JSON or file path (optional, defaults to stdin)",
    )

    args = parser.parse_args()

    if args.request_json:
        # File or JSON string provided as argument
        req_dict = _parse_request_json(args.request_json)
        # Check if stdin has additional params to merge
        stdin_params = _read_stdin_json_or_none()
        if stdin_params is not None:
            req_dict["params"] = _merge_params(req_dict.get("params"), stdin_params)
    else:
        # No argument provided, read full request from stdin
        stdin_data = sys.stdin.read()
        if not stdin_data.strip():
            print("Error: No request JSON provided via stdin or argument", file=sys.stderr)
            return 1
        try:
            req_dict = json.loads(stdin_data)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON from stdin: {e}", file=sys.stderr)
            return 1

    return asyncio.run(_run(req_dict))


if __name__ == "__main__":
    raise SystemExit(main())
