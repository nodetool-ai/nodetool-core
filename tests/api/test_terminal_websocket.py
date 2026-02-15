import json
import os
import sys
import time
from contextlib import contextmanager

import msgpack
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from nodetool.api.server import create_app

# Ensure all tests in this module run in the same xdist worker to prevent environment conflicts
# Use a dedicated group to avoid PTY operations crashing the worker when co-located with other tests
pytestmark = pytest.mark.xdist_group(name="terminal_ws")


@contextmanager
def _make_client(monkeypatch, env: str = "development", enable_flag: str | None = None):
    """Context manager for creating a test client with proper cleanup."""
    monkeypatch.setenv("ENV", env)
    # Some production paths require this to be set; use a benign value for tests.
    if env == "production":
        monkeypatch.setenv("SECRETS_MASTER_KEY", "test-key")
        # Set dummy S3 credentials and storage config for production mode tests
        monkeypatch.setenv("S3_ACCESS_KEY_ID", "test-s3-key")
        monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "test-s3-secret")
        monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:9000")
        monkeypatch.setenv("ASSET_TEMP_BUCKET", "test-temp-bucket")
        monkeypatch.setenv("ASSET_TEMP_DOMAIN", "http://localhost:9000")
    if enable_flag is None:
        monkeypatch.delenv("NODETOOL_ENABLE_TERMINAL_WS", raising=False)
    else:
        monkeypatch.setenv("NODETOOL_ENABLE_TERMINAL_WS", enable_flag)
    with TestClient(create_app()) as client:
        yield client


def test_terminal_ws_rejects_when_disabled(monkeypatch):
    with _make_client(monkeypatch) as client, client.websocket_connect("/terminal") as ws:
        msg = ws.receive()
        assert msg["type"] == "websocket.close"
        assert msg["code"] == 1008


def test_terminal_ws_rejects_in_production(monkeypatch):
    with (
        _make_client(monkeypatch, env="production", enable_flag="1") as client,
        client.websocket_connect("/terminal") as ws,
    ):
        msg = ws.receive()
        assert msg["type"] == "websocket.close"
        assert msg["code"] == 1008


@pytest.mark.skipif(sys.platform.startswith("win"), reason="PTY echo test assumes POSIX shell")
@pytest.mark.skipif(
    os.environ.get("PYTEST_XDIST_WORKER") is not None,
    reason="PTY-based terminal test crashes xdist workers; run without -n",
)
@pytest.mark.timeout(15)
def test_terminal_ws_echoes_input(monkeypatch):
    with _make_client(monkeypatch, enable_flag="1") as client, client.websocket_connect("/terminal") as ws:
        # Use text frames to keep JSON mode, and ensure the shell exits promptly.
        ws.send_json({"type": "input", "data": "echo hello && exit\n"})

        output_chunks: list[str] = []
        seen_hello = False

        for _ in range(30):
            try:
                msg = ws.receive()
            except WebSocketDisconnect:
                break

            if "bytes" in msg and msg["bytes"] is not None:
                payload = msgpack.unpackb(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                payload = json.loads(msg["text"])
            else:
                continue

            if payload.get("type") == "output":
                chunk = payload.get("data", "")
                output_chunks.append(chunk)
                if "hello" in chunk:
                    seen_hello = True
            if payload.get("type") == "exit":
                break

    output_text = "".join(output_chunks)
    assert seen_hello, f"Did not observe expected output. Collected: {output_text!r}"
