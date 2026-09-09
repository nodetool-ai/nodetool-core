from __future__ import annotations

from unittest.mock import patch

import pytest

from nodetool.io.http_fetch import fetch_http_bytes


class _Response:
    def __init__(self, status: int, *, location: str | None = None, data: bytes = b"") -> None:
        self.status = status
        self.headers = {} if location is None else {"Location": location}
        self._data = data

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def read(self) -> bytes:
        return self._data


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = iter(responses)
        self.requested: list[str] = []

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def get(self, uri: str, *, allow_redirects: bool) -> _Response:
        assert allow_redirects is False
        self.requested.append(uri)
        return next(self.responses)


@pytest.mark.asyncio
async def test_fetch_http_bytes_resolves_relative_redirects() -> None:
    session = _Session([_Response(302, location="/asset.bin"), _Response(200, data=b"ok")])
    with patch("nodetool.io.http_fetch.aiohttp.ClientSession", return_value=session):
        result = await fetch_http_bytes("https://example.com/start")

    assert result.data == b"ok"
    assert result.final_url == "https://example.com/asset.bin"
    assert session.requested == ["https://example.com/start", "https://example.com/asset.bin"]


@pytest.mark.asyncio
async def test_fetch_http_bytes_rejects_redirect_without_location() -> None:
    session = _Session([_Response(302)])
    with patch("nodetool.io.http_fetch.aiohttp.ClientSession", return_value=session):
        with pytest.raises(ValueError, match="without Location"):
            await fetch_http_bytes("https://example.com/start")


@pytest.mark.asyncio
async def test_fetch_http_bytes_rejects_private_redirect_target() -> None:
    session = _Session([_Response(302, location="http://127.0.0.1/private")])
    with patch("nodetool.io.http_fetch.aiohttp.ClientSession", return_value=session):
        with pytest.raises(ValueError, match="private/restricted"):
            await fetch_http_bytes("https://example.com/start")

    assert session.requested == ["https://example.com/start"]


@pytest.mark.asyncio
async def test_fetch_http_bytes_bounds_redirects() -> None:
    session = _Session([_Response(302, location=f"/hop/{index}") for index in range(6)])
    with patch("nodetool.io.http_fetch.aiohttp.ClientSession", return_value=session):
        with pytest.raises(ValueError, match="Too many redirects"):
            await fetch_http_bytes("https://example.com/start")

    assert len(session.requested) == 6
