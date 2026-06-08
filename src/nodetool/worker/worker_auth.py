"""Authentication for the WebSocket worker.

A shared-secret bearer token, opt-in via the ``NODETOOL_WORKER_TOKEN`` env var.

- Token unset/empty -> the worker is open (preserves local/stdio/dev behavior).
- Token set -> connections must present ``Authorization: Bearer <token>`` on the
  WebSocket opening handshake; the comparison is constant time.

The :func:`authorize` helper is pure (no env, no I/O) so it can be unit-tested in
isolation. :func:`make_process_request` builds the ``process_request`` hook that
:func:`nodetool.worker.server.start_server` wires into ``websockets`` ``serve``.
"""

from __future__ import annotations

import hmac
import http
import os
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection
    from websockets.http11 import Request, Response

BEARER_PREFIX = "Bearer "


def authorize(auth_header: str | None, expected_token: str | None) -> bool:
    """Decide whether a connection is authorized.

    Args:
        auth_header: The value of the ``Authorization`` request header, or
            :obj:`None` if the header is absent.
        expected_token: The configured worker token, or :obj:`None`/empty if no
            token is configured.

    Returns:
        ``True`` when the worker is open (no expected token) or when
        ``auth_header`` is exactly ``"Bearer <expected_token>"``. ``False``
        otherwise. The token comparison uses :func:`hmac.compare_digest` to avoid
        leaking the token through timing.
    """
    if not expected_token:
        return True
    if not auth_header:
        return False
    return hmac.compare_digest(auth_header, BEARER_PREFIX + expected_token)


def make_process_request() -> Callable[["ServerConnection", "Request"], "Response | None"]:
    """Build a ``process_request`` hook that enforces the worker token.

    The expected token is read from ``NODETOOL_WORKER_TOKEN`` on each handshake,
    so the gate reflects the current environment without a server restart.

    Returns:
        A function suitable for ``websockets`` ``serve(..., process_request=...)``.
        It returns :obj:`None` to allow the handshake, or an HTTP 401
        :class:`~websockets.http11.Response` to abort it before any frame.
    """

    def process_request(
        connection: "ServerConnection", request: "Request"
    ) -> "Response | None":
        expected_token = os.environ.get("NODETOOL_WORKER_TOKEN")
        if authorize(request.headers.get("Authorization"), expected_token):
            return None
        return connection.respond(
            http.HTTPStatus.UNAUTHORIZED,
            "Unauthorized\n",
        )

    return process_request
