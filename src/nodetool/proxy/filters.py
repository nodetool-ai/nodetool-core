"""
HTTP header filtering utilities for the async reverse proxy.

Filters hop-by-hop headers and other headers that should not be forwarded.
"""


# RFC 7230: Hop-by-hop headers that must not be forwarded
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",  # Let FastAPI/ASGI set this automatically
}

# Additional headers that should not be forwarded
EXCLUDED_HEADERS = HOP_BY_HOP_HEADERS | {
    "host",  # Always set based on upstream server
}


def filter_headers(headers: dict[str, str], exclude: set[str] | None = None) -> dict[str, str]:
    """
    Filter out hop-by-hop and other excluded headers.

    Args:
        headers: Headers to filter.
        exclude: Additional headers to exclude (defaults to EXCLUDED_HEADERS).

    Returns:
        Filtered headers dictionary.
    """
    if exclude is None:
        exclude = EXCLUDED_HEADERS

    return {k: v for k, v in headers.items() if k.lower() not in exclude}


def filter_request_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Filter request headers for forwarding to upstream.

    Removes hop-by-hop headers and Host header.

    Args:
        headers: Request headers.

    Returns:
        Filtered headers suitable for upstream forwarding.
    """
    return filter_headers(headers)


def filter_response_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Filter response headers for returning to client.

    Removes hop-by-hop headers (except Content-Length which ASGI handles).

    Args:
        headers: Response headers from upstream.

    Returns:
        Filtered headers suitable for client response.
    """
    return filter_headers(headers)
