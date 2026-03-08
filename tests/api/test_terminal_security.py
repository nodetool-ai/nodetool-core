from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocket

from nodetool.api.server import create_app


@pytest.mark.asyncio
async def test_terminal_rejects_external_ip_when_auth_disabled(monkeypatch):
    monkeypatch.setenv("NODETOOL_ENABLE_TERMINAL_WS", "1")
    monkeypatch.setenv("AUTH_PROVIDER", "local")

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
    # Important: The mock must have the attributes expected by the endpoint
    mock_ws.client = MagicMock()
    mock_ws.client.host = "192.168.1.5"  # External IP

    # The endpoint is an async function (or coroutine function)
    endpoint = terminal_route.endpoint

    # Call the endpoint
    # Note: The endpoint expects a WebSocket instance.
    await endpoint(mock_ws)

    # Verify calls
    # It should have accepted and then closed with 1008
    mock_ws.accept.assert_called_once()
    # The close reason must match exactly what is in server.py
    mock_ws.close.assert_called_with(code=1008, reason="External terminal access requires configured authentication.")
