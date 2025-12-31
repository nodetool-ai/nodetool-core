from __future__ import annotations

import logging
from io import BytesIO
from typing import Tuple

import aiohttp
import httpx
import numpy as np
import PIL.Image

from nodetool.config.environment import Environment
from nodetool.media.image.image_utils import (
    numpy_to_pil_image,
    pil_to_png_bytes,
)
from nodetool.runtime.resources import require_scope

log = logging.getLogger(__name__)


def _normalize_image_like_to_png_bytes(obj: object) -> bytes:
    """
    Best-effort conversion of common image-like objects to PNG bytes.
    Supports PIL.Image.Image, numpy.ndarray, raw bytes (if image), and file-like.
    Raises ValueError if the object cannot be converted.
    """
    if isinstance(obj, PIL.Image.Image):
        return pil_to_png_bytes(obj)

    if isinstance(obj, np.ndarray):
        pil_img = numpy_to_pil_image(obj)
        return pil_to_png_bytes(pil_img)

    if isinstance(obj, bytes | bytearray):
        raw = bytes(obj)
        try:
            with PIL.Image.open(BytesIO(raw)) as img:
                return pil_to_png_bytes(img)
        except Exception as e:
            raise ValueError(f"Bytes are not a decodable image: {e}") from e

    if hasattr(obj, "read"):
        try:
            pos = obj.tell() if hasattr(obj, "tell") else None
            if hasattr(obj, "seek"):
                obj.seek(0)
            raw = obj.read()
            if pos is not None and hasattr(obj, "seek"):
                obj.seek(pos)
            with PIL.Image.open(BytesIO(raw)) as img:
                return pil_to_png_bytes(img)
        except Exception as e:
            raise ValueError(f"File-like object could not be decoded as image: {e}") from e

    raise ValueError(f"Unsupported object type for image conversion: {type(obj)}")


def _parse_data_uri(uri: str) -> Tuple[str, bytes]:
    """Parse a data: URI and return (mime, bytes)."""
    try:
        header, b64data = uri.split(",", 1)
        mime_type = "application/octet-stream"
        if ";" in header:
            meta = header[5:]
            parts = meta.split(";")
            if parts and "/" in parts[0]:
                mime_type = parts[0]
        import base64

        raw = base64.b64decode(b64data.encode("utf-8") if not isinstance(b64data, bytes) else b64data)
        return mime_type, raw
    except Exception as e:
        # Tests expect the phrase "Invalid data URI" to appear
        raise ValueError(f"Invalid data URI: {e}") from e


def _fetch_file_uri(uri: str) -> Tuple[str, bytes]:
    """Read file:// URI and return (mime, bytes).

    Use builtins.open so tests can mock file IO with patch("builtins.open").
    """
    import mimetypes
    import pathlib

    path = uri[len("file://") :]
    # Normalize path
    p = pathlib.Path(path)
    with open(p, "rb") as f:  # type: ignore[arg-type]
        data = f.read()
    mime_type, _ = mimetypes.guess_type(uri)
    if not mime_type:
        mime_type = "application/octet-stream"
    return mime_type, data


async def _fetch_http_uri_async(uri: str) -> Tuple[str, bytes]:
    """Fetch content from an HTTP/HTTPS URL. Local storage URLs are handled by the caller."""
    async with aiohttp.ClientSession() as session, session.get(uri) as response:
        response.raise_for_status()
        data = await response.read()
        content_type = response.headers.get("Content-Type")
        mime_type: str | None = None
        if content_type:
            mime_type = content_type.split(";", 1)[0]
        if not mime_type:
            import mimetypes

            mime_type, _ = mimetypes.guess_type(uri)
        if not mime_type:
            mime_type = "application/octet-stream"
        return mime_type, data


def _is_local_storage_url(uri: str) -> bool:
    """Check if this is a local storage URL that should be read directly from storage."""
    import re

    # Match localhost or 127.0.0.1 with any port, accessing /api/storage/
    pattern = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?/api/storage/"
    return bool(re.match(pattern, uri))


def _extract_storage_key_from_url(uri: str) -> str:
    """Extract the storage key from a local storage URL."""
    import re

    # Extract the key from URLs like: http://localhost:7777/api/storage/828ae5ded94411f0884a000022ae8b15.png
    match = re.search(r"/api/storage/(.+)$", uri)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract storage key from URL: {uri}")


def _fetch_local_storage_sync(uri: str) -> Tuple[str, bytes]:
    """Read directly from local storage instead of making HTTP request.

    This function works correctly whether called from a sync or async context
    by running the async storage operations in a separate thread.
    """
    import asyncio
    import concurrent.futures
    import mimetypes

    key = _extract_storage_key_from_url(uri)

    # Get storage from the current scope
    scope = require_scope()
    storage = scope.get_asset_storage()

    async def _do_fetch() -> bytes:
        """Helper to run async storage operations."""
        exists = await storage.file_exists(key)
        if not exists:
            raise ValueError(f"Storage file not found: {key}")

        stream = BytesIO()
        await storage.download(key, stream)
        return stream.getvalue()

    # Run the async code in a separate thread to avoid event loop conflicts
    # when this sync function is called from an async context
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, _do_fetch())
        data: bytes = future.result()

    # Guess mime type from the key
    mime_type, _ = mimetypes.guess_type(key)
    if not mime_type:
        mime_type = "application/octet-stream"

    return mime_type, data


async def _fetch_local_storage_async(uri: str) -> Tuple[str, bytes]:
    """Read directly from local storage instead of making HTTP request (async version)."""
    import mimetypes

    key = _extract_storage_key_from_url(uri)

    # Get storage from the current scope
    scope = require_scope()
    storage = scope.get_asset_storage()

    exists = await storage.file_exists(key)
    if not exists:
        raise ValueError(f"Storage file not found: {key}")

    # Download the file
    stream = BytesIO()
    await storage.download(key, stream)
    data = stream.getvalue()

    # Guess mime type from the key
    mime_type, _ = mimetypes.guess_type(key)
    if not mime_type:
        mime_type = "application/octet-stream"

    return mime_type, data


def _fetch_http_uri_sync(uri: str) -> Tuple[str, bytes]:
    # Check for local storage URLs first - read directly instead of HTTP call
    if _is_local_storage_url(uri):
        return _fetch_local_storage_sync(uri)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = httpx.get(uri, headers=headers, follow_redirects=True)
    resp.raise_for_status()
    data = resp.content
    content_type = resp.headers.get("Content-Type")
    mime_type: str | None = None
    if content_type:
        mime_type = content_type.split(";", 1)[0]
    if not mime_type:
        import mimetypes

        mime_type, _ = mimetypes.guess_type(uri)
    if not mime_type:
        mime_type = "application/octet-stream"
    return mime_type, data


def _fetch_memory_uri(uri: str) -> Tuple[str, bytes]:
    obj = None
    try:
        obj = require_scope().get_memory_uri_cache().get(uri)
    except Exception as e:
        log.debug(f"Failed to get from memory URI cache: {e}")
    if obj is None:
        raise ValueError(f"No cached object for memory URI: {uri}")

    # Prefer image normalization when possible
    try:
        data = _normalize_image_like_to_png_bytes(obj)
        return "image/png", data
    except Exception as e:
        log.debug(f"Failed to normalize image: {e}")

    # Fallbacks
    if isinstance(obj, bytes | bytearray):
        return "application/octet-stream", bytes(obj)

    raise ValueError(f"Unsupported object type for memory URI: {type(obj)}")


async def fetch_uri_bytes_and_mime_async(uri: str) -> Tuple[str, bytes]:
    """
    Fetch content from a URI and return (mime_type, data_bytes).
    Supports data:, memory://, file://, and http(s) URIs.
    """
    if uri.startswith("data:"):
        return _parse_data_uri(uri)
    if uri.startswith("memory://"):
        return _fetch_memory_uri(uri)
    if uri.startswith("file://"):
        return _fetch_file_uri(uri)
    if uri.startswith("http://") or uri.startswith("https://"):
        # Check for local storage URLs first - read directly instead of HTTP call
        if _is_local_storage_url(uri):
            return await _fetch_local_storage_async(uri)
        return await _fetch_http_uri_async(uri)
    # Explicitly reject unsupported schemes (e.g., ftp)
    raise ValueError(f"Unsupported URI scheme: {uri.split(':', 1)[0]}://")


def fetch_uri_bytes_and_mime_sync(uri: str) -> Tuple[str, bytes]:
    """
    Synchronous variant of fetch_uri_bytes_and_mime_async using httpx for http(s).
    """
    if uri.startswith("data:"):
        return _parse_data_uri(uri)
    if uri.startswith("memory://"):
        return _fetch_memory_uri(uri)
    if uri.startswith("file://"):
        return _fetch_file_uri(uri)
    if uri.startswith("http://") or uri.startswith("https://"):
        return _fetch_http_uri_sync(uri)
    # Explicitly reject unsupported schemes (e.g., ftp)
    raise ValueError(f"Unsupported URI scheme: {uri.split(':', 1)[0]}://")
