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


async def fetch_http_bytes(uri: str, headers: dict[str, str] | None = None) -> HTTPFetchResult:
    """Fetch HTTP bytes with SSRF checks applied to every redirect hop.

    Redirects are intentionally handled here instead of by aiohttp so a
    redirect target is checked before it is requested.  The resolver also
    checks DNS results, including hosts that resolve to mixed public/private
    addresses.
    """

    current_headers = dict(headers) if headers else {}
    current_uri = uri
    connector = aiohttp.TCPConnector(resolver=SSRFProtectResolver())
    async with aiohttp.ClientSession(connector=connector) as session:
        for _ in range(MAX_REDIRECTS + 1):
            parsed = urlparse(current_uri)
            hostname = parsed.hostname
            if hostname and is_ip_private(hostname):
                raise ValueError(f"Access to private/restricted IP blocked: {hostname}")

            async with session.get(current_uri, headers=current_headers, allow_redirects=False) as response:
                if response.status in REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if not location:
                        raise HTTPRedirectError(
                            f"Redirect response without Location header from {current_uri}",
                            url=current_uri,
                        )
                    current_uri = urljoin(current_uri, location)
                    parsed_current = urlparse(current_uri)
                    if (parsed_current.scheme, parsed_current.netloc) != (parsed.scheme, parsed.netloc):
                        keys_to_remove = [k for k in current_headers if k.lower() == "authorization"]
                        for k in keys_to_remove:
                            current_headers.pop(k, None)
                    continue

                response.raise_for_status()
                return HTTPFetchResult(
                    data=await response.read(),
                    final_url=current_uri,
                    content_type=response.headers.get("Content-Type"),
                )

    raise HTTPTooManyRedirects(f"Too many redirects while fetching {uri}")
