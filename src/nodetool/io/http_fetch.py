"""Guarded, manually redirected HTTP byte fetching."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import aiohttp

from nodetool.utils.network import SSRFProtectResolver, is_ip_private

REDIRECT_STATUSES = (301, 302, 303, 307, 308)
MAX_REDIRECTS = 5


class HTTPRedirectError(ValueError):
    """A redirect response could not be followed safely."""

    def __init__(self, message: str, *, url: str) -> None:
        super().__init__(message)
        self.url = url


class HTTPTooManyRedirects(ValueError):
    """The configured redirect hop limit was reached."""


@dataclass(frozen=True)
class HTTPFetchResult:
    """Bytes returned by a guarded request and the URL that served them."""

    data: bytes
    final_url: str
    content_type: str | None


async def fetch_http_bytes(uri: str) -> HTTPFetchResult:
    """Fetch HTTP bytes with SSRF checks applied to every redirect hop.

    Redirects are intentionally handled here instead of by aiohttp so a
    redirect target is checked before it is requested.  The resolver also
    checks DNS results, including hosts that resolve to mixed public/private
    addresses.
    """

    current_uri = uri
    connector = aiohttp.TCPConnector(resolver=SSRFProtectResolver())
    async with aiohttp.ClientSession(connector=connector) as session:
        for _ in range(MAX_REDIRECTS + 1):
            hostname = urlparse(current_uri).hostname
            if hostname and is_ip_private(hostname):
                raise ValueError(f"Access to private/restricted IP blocked: {hostname}")

            async with session.get(current_uri, allow_redirects=False) as response:
                if response.status in REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if not location:
                        raise HTTPRedirectError(
                            f"Redirect response without Location header from {current_uri}",
                            url=current_uri,
                        )
                    current_uri = urljoin(current_uri, location)
                    continue

                response.raise_for_status()
                return HTTPFetchResult(
                    data=await response.read(),
                    final_url=current_uri,
                    content_type=response.headers.get("Content-Type"),
                )

    raise HTTPTooManyRedirects(f"Too many redirects while fetching {uri}")
