
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocket

from nodetool.api.server import create_app
from nodetool.config.environment import Environment


@pytest.mark.asyncio
async def test_terminal_rejects_evil_origin_when_auth_disabled(monkeypatch):
    monkeypatch.setenv("NODETOOL_ENABLE_TERMINAL_WS", "1")
    monkeypatch.setenv("AUTH_PROVIDER", "local")

    # Ensure auth is disabled
    monkeypatch.setattr(Environment, "enforce_auth", lambda: False)

    app = create_app()

    # Find the websocket route
    terminal_route = None
    for route in app.routes:
        if getattr(route, "path", "") == "/ws/terminal":
            terminal_route = route
            break

    assert terminal_route is not None, "Terminal websocket route not found"

    # Mock WebSocket
    mock_ws = AsyncMock(spec=WebSocket)
    # Simulate connection from localhost
    mock_ws.client = MagicMock()
    mock_ws.client.host = "127.0.0.1"

    # Simulate evil origin
    mock_ws.headers = {"origin": "http://evil.com"}

    # Mock TerminalWebSocketRunner
    mock_runner_instance = AsyncMock()
    mock_runner_class = MagicMock(return_value=mock_runner_instance)

    monkeypatch.setattr("nodetool.integrations.websocket.terminal_runner.TerminalWebSocketRunner", mock_runner_class)
    monkeypatch.setattr("nodetool.integrations.websocket.terminal_runner.TerminalWebSocketRunner.is_enabled", lambda: True)

    endpoint = terminal_route.endpoint
    await endpoint(mock_ws)

    # Verify the connection was rejected
    assert mock_ws.close.called, "Connection should have been rejected due to invalid origin"
    _, kwargs = mock_ws.close.call_args
    assert kwargs.get('code') == 1008
    reason = kwargs.get('reason', "").lower()
    assert "origin" in reason or "external" in reason # My fix uses specific reason for origin

@pytest.mark.asyncio
async def test_terminal_allows_localhost_origin(monkeypatch):
    monkeypatch.setenv("NODETOOL_ENABLE_TERMINAL_WS", "1")
    monkeypatch.setenv("AUTH_PROVIDER", "local")

    # Ensure auth is disabled
    monkeypatch.setattr(Environment, "enforce_auth", lambda: False)

    app = create_app()

    # Find the websocket route
    terminal_route = None
    for route in app.routes:
        if getattr(route, "path", "") == "/ws/terminal":
            terminal_route = route
            break

    # Mock WebSocket
    mock_ws = AsyncMock(spec=WebSocket)
    mock_ws.client = MagicMock()
    mock_ws.client.host = "127.0.0.1"

    # Simulate valid origin
    mock_ws.headers = {"origin": "http://localhost:8000"}

    # Mock TerminalWebSocketRunner
    mock_runner_instance = AsyncMock()
    mock_runner_class = MagicMock(return_value=mock_runner_instance)

    monkeypatch.setattr("nodetool.integrations.websocket.terminal_runner.TerminalWebSocketRunner", mock_runner_class)
    monkeypatch.setattr("nodetool.integrations.websocket.terminal_runner.TerminalWebSocketRunner.is_enabled", lambda: True)

    endpoint = terminal_route.endpoint
    await endpoint(mock_ws)

    # Verify connection allowed
    assert not mock_ws.close.called, "Connection should be allowed for localhost origin"
    mock_runner_instance.run.assert_called_once()

@pytest.mark.asyncio
async def test_terminal_allows_missing_origin(monkeypatch):
    monkeypatch.setenv("NODETOOL_ENABLE_TERMINAL_WS", "1")
    monkeypatch.setenv("AUTH_PROVIDER", "local")

    # Ensure auth is disabled
    monkeypatch.setattr(Environment, "enforce_auth", lambda: False)

    app = create_app()

    # Find the websocket route
    terminal_route = None
    for route in app.routes:
        if getattr(route, "path", "") == "/ws/terminal":
            terminal_route = route
            break

    # Mock WebSocket
    mock_ws = AsyncMock(spec=WebSocket)
    mock_ws.client = MagicMock()
    mock_ws.client.host = "127.0.0.1"

    # Simulate missing origin (CLI tool)
    mock_ws.headers = {}

    # Mock TerminalWebSocketRunner
    mock_runner_instance = AsyncMock()
    mock_runner_class = MagicMock(return_value=mock_runner_instance)

    monkeypatch.setattr("nodetool.integrations.websocket.terminal_runner.TerminalWebSocketRunner", mock_runner_class)
    monkeypatch.setattr("nodetool.integrations.websocket.terminal_runner.TerminalWebSocketRunner.is_enabled", lambda: True)

    endpoint = terminal_route.endpoint
    await endpoint(mock_ws)

    # Verify connection allowed
    assert not mock_ws.close.called, "Connection should be allowed for missing origin"
    mock_runner_instance.run.assert_called_once()
