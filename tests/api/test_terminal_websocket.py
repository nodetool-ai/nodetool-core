import json
import os
import sys
import time

import msgpack

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from nodetool.api.server import create_app


def _make_client(monkeypatch, env: str = "development", enable_flag: str | None = None) -> TestClient:
    monkeypatch.setenv("ENV", env)
    # Some production paths require this to be set; use a benign value for tests.
    if env == "production":
        monkeypatch.setenv("SECRETS_MASTER_KEY", "test-key")
    if enable_flag is None:
        monkeypatch.delenv("NODETOOL_ENABLE_TERMINAL_WS", raising=False)
    else:
        monkeypatch.setenv("NODETOOL_ENABLE_TERMINAL_WS", enable_flag)
    return TestClient(create_app())


def test_terminal_ws_rejects_when_disabled(monkeypatch):
    with _make_client(monkeypatch) as client:
        with client.websocket_connect("/terminal") as ws:
            msg = ws.receive()
            assert msg["type"] == "websocket.close"
            assert msg["code"] == 1008


def test_terminal_ws_rejects_in_production(monkeypatch):
    with _make_client(monkeypatch, env="production", enable_flag="1") as client:
        with client.websocket_connect("/terminal") as ws:
            msg = ws.receive()
            assert msg["type"] == "websocket.close"
            assert msg["code"] == 1008


@pytest.mark.skipif(sys.platform.startswith("win"), reason="PTY echo test assumes POSIX shell")
@pytest.mark.timeout(15)
def test_terminal_ws_echoes_input(monkeypatch):
    with _make_client(monkeypatch, enable_flag="1") as client:
        with client.websocket_connect("/terminal") as ws:
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
